"""LinearRegression branch — 短期反转预测 (mean reversion).

设计:
  - Tree (LightGBM): 187 dim 因子特征 (97 Tier1 + 89 Tier2 + 1 Tier3=industry_id)
  - LinearRegression: 10 dim **反转** features (无 RSI, 跟 Tree 互补不重复)
  - 集成: Stack Model (LightGBM with 5-6 features per regime)

为什么 Linear 负责"反转预测"而不是"动量延续":
  数据集分析 (Spearman 相关性, 全 11 年):
    - 5d 持有期, 1d~60d lag 全部**负相关** (-0.008 ~ -0.031)
    - 所有 11 行业、所有 HS300 regime 都反转
    - Decile 分析: 5d lag top-bot = -0.0015 (REV)
  A 股是**反转市场** (contrarian), 不是动量市场 (momentum).

Linear 10 维 features:
  - 4 个反转 return: -ret_1d, -ret_3d, -ret_5d, -ret_10d
  - 2 个 MA distance: dist_ma5, dist_ma20 (偏离均线, mean reversion)
  - 1 个 BOLL position: boll_pos (布林带位置)
  - 2 个 streak: up_streak, down_streak (连续涨跌)
  - 1 个 range_pct_5d (反转时波动大)

Input: 10 features (all reversal)
Output: per-stock 5d return score
"""
import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore', message='Workbook contains no default style')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from model import LinearRegressionModel
from industry_mapping import load_industry_map
from paths import INTEGRATED_DIR, MODEL_INTEGRATED_DIR, TRAIN_CSV

# LinearRegression 10 维**反转** features (mean reversion)
# 数据集分析结论: A 股 5d 持有期是反转市场, 动量延续不成立
# 设计原则:
#   - 4 个反转 return: -ret_1d, -ret_3d, -ret_5d, -ret_10d
#   - 2 个 MA distance: 偏离均线 (mean reversion)
#   - 1 个 BOLL position
#   - 2 个 streak (reversal signal)
#   - 1 个 range_pct_5d (反转时波动大)
# Tree 用 187 dim 因子 (含 RSI), Linear 用 10 dim 纯反转 (无 RSI), 互补不重复.
REVERSAL_FEATURES = [
    'neg_ret_1d',      # -ret_1d (反转)
    'neg_ret_3d',      # -ret_3d (反转)
    'neg_ret_5d',      # -ret_5d (反转, 最强)
    'neg_ret_10d',     # -ret_10d (反转)
    'dist_ma5',        # (close - MA5) / MA5
    'dist_ma20',       # (close - MA20) / MA20
    'boll_pos',        # 布林带位置 [0, 1]
    'up_streak',       # 连续上涨天数 / 5
    'down_streak',     # 连续下跌天数 / 5
    'range_pct_5d',    # 5d 高低范围 / close (反转时波动大)
]


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_reversal_features(df):
    """构建 10 维**反转** features (mean reversion).

    设计依据: 数据集 Spearman 分析显示 5d 持有期是反转市场 (rho < 0)
    所有 lag (1d/3d/5d/10d/20d/60d) 都跟 forward 5d 收益负相关.
    故 features 显式捕捉反转信号 (MA 偏离, BOLL 位置, streak).

    Input df 必须按 股票代码, 日期 排序, 含 OHLCV 列.
    输出在原 df 上加 10 列 (REVERSAL_FEATURES).
    """
    df = df.copy()
    g = df.groupby('股票代码', sort=False)
    close = df['收盘']
    high = df['最高']
    low = df['最低']

    # 1-4. 4 个反转 return (取负, 显式反转)
    for lag in [1, 3, 5, 10]:
        df[f'neg_ret_{lag}d'] = -(close / g['收盘'].shift(lag) - 1)

    # 删 RSI 6d/14d (跟 Tree Tier1.rsi 重复, 抽象层次重叠)
    # 改用 dist_ma + boll_pos 替代反转信号

    # 5. dist_ma5: (close - MA5) / MA5
    ma5 = g['收盘'].rolling(5, min_periods=2).mean().reset_index(level=0, drop=True)
    df['dist_ma5'] = ((close - ma5) / (ma5 + 1e-9)).clip(-0.2, 0.2)

    # 8. dist_ma20
    ma20 = g['收盘'].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    df['dist_ma20'] = ((close - ma20) / (ma20 + 1e-9)).clip(-0.3, 0.3)

    # 9. boll_pos: (close - MA20) / (2 * std20) → [-1, 1]
    std20 = g['收盘'].rolling(20, min_periods=5).std().reset_index(level=0, drop=True)
    df['boll_pos'] = ((close - ma20) / (2 * std20 + 1e-9)).clip(-1, 1) * 0.5 + 0.5  # [0, 1]

    # 10. up_streak: 连续上涨天数 / 5
    ret_1d = close / g['收盘'].shift(1) - 1
    up_1d = (ret_1d > 0).astype(int)
    df['up_streak'] = (up_1d.groupby(df['股票代码']).rolling(5, min_periods=1).sum()
                       .reset_index(level=0, drop=True) / 5.0)

    # 11. down_streak: 连续下跌天数 / 5
    down_1d = (ret_1d < 0).astype(int)
    df['down_streak'] = (down_1d.groupby(df['股票代码']).rolling(5, min_periods=1).sum()
                          .reset_index(level=0, drop=True) / 5.0)

    # 12. range_pct_5d: 5d (high - low) / close
    high_5 = g['最高'].rolling(5, min_periods=2).max().reset_index(level=0, drop=True)
    low_5 = g['最低'].rolling(5, min_periods=2).min().reset_index(level=0, drop=True)
    df['range_pct_5d'] = ((high_5 - low_5) / (close + 1e-9)).clip(0, 0.3)

    # 清理 inf/nan
    for c in REVERSAL_FEATURES:
        df[c] = df[c].replace([np.inf, -np.inf], 0).fillna(0)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=str(MODEL_INTEGRATED_DIR / 'linear_regression'),
                        help='模型权重目录 (best_seed*.pth)')
    parser.add_argument('--output_dir', type=str, default=str(INTEGRATED_DIR / 'linear_regression'),
                        help='推理产物目录 (oof_scores, val_meta, scaler 等)')
    parser.add_argument('--val_meta_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='L2 regularization (Ridge-like, 加大防发散)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n{'=' * 80}\n>>> [LinearRegression] starting (10 dim 反转 features)\n", flush=True)
    print(f"[Config] model_dir={args.model_dir}", flush=True)
    print(f"[Config] output_dir={args.output_dir}, epochs={args.epochs}, lr={args.lr}, "
          f"weight_decay={args.weight_decay}, seeds={args.seeds}, smoke={args.smoke}", flush=True)

    # 1. Load data
    print("\n[Data] loading stock_data.csv ...", flush=True)
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    print(f"[Data] {len(df):,} rows, {df['股票代码'].nunique()} stocks, "
          f"{df['日期'].min().date()} ~ {df['日期'].max().date()}", flush=True)

    if args.smoke:
        sample_stocks = sorted(df['股票代码'].unique())[:30]
        df = df[df['股票代码'].isin(sample_stocks)].copy()
        all_dates = sorted(df['日期'].unique())
        df = df[df['日期'].isin(all_dates[-500:])].copy()
        print(f"[Smoke] {len(df):,} rows", flush=True)

    # 2. Build label (5d forward return, 与 Tree 一致)
    print("\n[Label] building 5d forward return ...", flush=True)
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['label'] = ((df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)).clip(-0.3, 0.3)
    df = df.dropna(subset=['label'])

    # 3. 10 维**反转** features (mean reversion, 数据集分析显示 A 股是反转市场)
    print("\n[Features] building 10 反转 features (MA/BOLL/streak) ...", flush=True)
    processed = build_reversal_features(df)
    for c in REVERSAL_FEATURES:
        print(f"  {c:20s}: range=[{processed[c].min():.4f}, "
              f"{processed[c].max():.4f}], mean={processed[c].mean():.4f}", flush=True)

    # Linear 不用 macro features (10 dim 纯反转, 跟 Macro 抽象层次分离)
    # Linear 负责 stock-level 短期反转, Macro 负责 industry-level alpha+beta
    # Stack Model 负责集成, 不需要在 Linear 内部重复 macro 信息
    print("\n[Features] Linear 10 维纯反转, 不含 macro (跟 Macro 职责分离)", flush=True)
    #    数据集分析: A 股 5d 持有期是反转市场 (rho=-0.031), 不是动量市场
    #    Linear 显式捕捉反转信号: -ret_*, MA distance, BOLL, streak
    feature_cols = REVERSAL_FEATURES
    feature_cols = [c for c in feature_cols if c in processed.columns]
    print(f"\n[Features] {len(feature_cols)} features (10 reversal, no macro): {feature_cols}",
          flush=True)

    # 6. Train/val/test split (3 段)
    val_start = pd.Timestamp(config['val_start'])
    val_end = pd.Timestamp(config['val_end'])
    test_start = pd.Timestamp(config['test_start'])
    test_end = pd.Timestamp(config['test_end'])
    train_mask = processed['日期'] < val_start
    val_mask = (processed['日期'] >= val_start) & (processed['日期'] <= val_end)
    test_mask = (processed['日期'] >= test_start) & (processed['日期'] <= test_end)
    print(f"[Split] train={train_mask.sum():,}, val={val_mask.sum():,}, "
          f"test={test_mask.sum():,}", flush=True)

    # 7. Build tensors
    X_train = processed.loc[train_mask, feature_cols].values.astype(np.float32)
    y_train = processed.loc[train_mask, 'label'].values.astype(np.float32)
    X_val = processed.loc[val_mask, feature_cols].values.astype(np.float32)
    y_val = processed.loc[val_mask, 'label'].values.astype(np.float32)
    X_test = processed.loc[test_mask, feature_cols].values.astype(np.float32)
    y_test = processed.loc[test_mask, 'label'].values.astype(np.float32)
    print(f"[Tensors] X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}", flush=True)

    # 8. Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    import joblib
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(args.output_dir, 'feature_names.pkl'))

    # 9. Load val_meta for OOF reshape
    val_meta_paths = []
    if args.val_meta_path:
        val_meta_paths.append(args.val_meta_path)
    for p in [os.path.join(os.path.dirname(args.output_dir), 'macro_transformer', 'val_meta.json'),
              str(INTEGRATED_DIR / 'macro_transformer' / 'val_meta.json')]:
        val_meta_paths.append(p)
    val_meta = None
    for p in val_meta_paths:
        if os.path.exists(p):
            with open(p) as f:
                val_meta = json.load(f)
            print(f"[ValMeta] loaded from {p}", flush=True)
            break
    if val_meta is None:
        print("[ValMeta] WARN: no val_meta.json found", flush=True)
        val_meta = {'dates': sorted(processed.loc[val_mask, '日期'].dt.strftime('%Y-%m-%d').unique().tolist())}

    # Load test_meta (3 月 test 段, 来自 macro_transformer)
    test_meta_paths = [
        os.path.join(os.path.dirname(args.output_dir), 'macro_transformer', 'test_meta.json'),
        str(INTEGRATED_DIR / 'macro_transformer' / 'test_meta.json'),
    ]
    test_meta = None
    for p in test_meta_paths:
        if os.path.exists(p):
            with open(p) as f:
                test_meta = json.load(f)
            print(f"[TestMeta] loaded from {p}", flush=True)
            break
    if test_meta is None:
        print("[TestMeta] WARN: no test_meta.json found", flush=True)
        test_meta = {'dates': sorted(processed.loc[test_mask, '日期'].dt.strftime('%Y-%m-%d').unique().tolist())}

    val_target_dates = set(val_meta['dates'])

    # 10. Train (8 维 features, 训练更快)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}", flush=True)

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n========== Seed {seed} ==========", flush=True)
        # output_clip=0.3 防止数值发散
        model = LinearRegressionModel(n_features=len(feature_cols), output_clip=0.3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
        huber = torch.nn.HuberLoss(delta=0.1)  # 比 MSE 抗 outlier

        X_train_t = torch.from_numpy(X_train).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().to(device)
        X_val_t = torch.from_numpy(X_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).float().to(device)
        X_test_t = torch.from_numpy(X_test).float().to(device)
        y_test_t = torch.from_numpy(y_test).float().to(device)

        B = 4096
        best_val_loss = float('inf')
        best_state = None
        for epoch in range(args.epochs):
            t0 = time.time()
            model.train()
            perm = torch.randperm(len(X_train_t))
            tr_loss = 0.0
            n_batch = 0
            for i in range(0, len(perm), B):
                idx = perm[i:i + B]
                pred = model(X_train_t[idx])
                loss = huber(pred, y_train_t[idx])  # Huber 比 MSE 抗 outlier
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                n_batch += 1
            tr_loss /= max(n_batch, 1)

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = ((val_pred - y_val_t) ** 2).mean().item()
                from scipy.stats import spearmanr
                ic, _ = spearmanr(val_pred.cpu().numpy(), y_val)
                ic = ic if not np.isnan(ic) else 0.0

            print(f"[seed={seed} epoch={epoch+1}/{args.epochs}] "
                  f"train Huber={tr_loss:.5f} | val MSE={val_loss:.5f} IC={ic:.4f} | "
                  f"{time.time()-t0:.1f}s", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(args.model_dir, f'best_seed{seed}.pth'))

        # 11. OOF on val + train + test
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            train_pred_all = model(X_train_t).cpu().numpy()
            val_pred_all = model(X_val_t).cpu().numpy()
            test_pred_all = model(X_test_t).cpu().numpy()

        # Common stock index for all OOFs
        all_stocks_full = sorted(pd.read_csv(TRAIN_CSV, encoding='utf-8-sig',
                                              usecols=['股票代码'])['股票代码'].unique())
        stockid2idx = {s: i for i, s in enumerate(all_stocks_full)}

        # train OOF (用于 stack model 训练 — 在 TRAIN 上训 stack)
        train_dates_arr = sorted(processed.loc[train_mask, '日期'].unique())
        train_dates_arr = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10] for d in train_dates_arr]
        train_date_to_idx = {d: i for i, d in enumerate(train_dates_arr)}
        train_oof = np.full((len(train_dates_arr), len(all_stocks_full)), np.nan, dtype=np.float32)
        train_proc = processed.loc[train_mask].copy()
        train_proc['_d_str'] = train_proc['日期'].dt.strftime('%Y-%m-%d')
        train_proc['_d_idx'] = train_proc['_d_str'].map(train_date_to_idx)
        train_proc['_s_idx'] = train_proc['股票代码'].map(stockid2idx)
        train_valid = train_proc['_d_idx'].notna() & train_proc['_s_idx'].notna()
        train_oof[train_proc.loc[train_valid, '_d_idx'].astype(int).values,
                  train_proc.loc[train_valid, '_s_idx'].astype(int).values] = train_pred_all[train_valid.values]
        np.save(os.path.join(args.output_dir, f'train_oof_seed{seed}.npy'), train_oof)
        print(f"[Save] train_oof_seed{seed}.npy: {train_oof.shape}, NaN frac={np.isnan(train_oof).mean():.3f}", flush=True)
        with open(os.path.join(args.output_dir, 'train_meta.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'dates': train_dates_arr,
                'stock_indices': [list(range(len(all_stocks_full))) for _ in train_dates_arr],
                'n_stocks': len(all_stocks_full),
                'seq_len': 60,
            }, f, ensure_ascii=False, indent=2)

        # val OOF
        val_dates_arr = sorted(val_target_dates)
        val_date_to_idx = {d: i for i, d in enumerate(val_dates_arr)}
        val_oof = np.full((len(val_dates_arr), len(all_stocks_full)), np.nan, dtype=np.float32)
        val_proc = processed.loc[val_mask].copy()
        val_proc['_d_str'] = val_proc['日期'].dt.strftime('%Y-%m-%d')
        val_proc['_d_idx'] = val_proc['_d_str'].map(val_date_to_idx)
        val_proc['_s_idx'] = val_proc['股票代码'].map(stockid2idx)
        val_valid = val_proc['_d_idx'].notna() & val_proc['_s_idx'].notna()
        val_oof[val_proc.loc[val_valid, '_d_idx'].astype(int).values,
                val_proc.loc[val_valid, '_s_idx'].astype(int).values] = val_pred_all[val_valid.values]
        np.save(os.path.join(args.output_dir, f'val_oof_seed{seed}.npy'), val_oof)
        with open(os.path.join(args.output_dir, 'val_meta.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'dates': val_dates_arr,
                'stock_indices': [list(range(len(all_stocks_full))) for _ in val_dates_arr],
                'n_stocks': len(all_stocks_full),
                'seq_len': 60,
            }, f, ensure_ascii=False, indent=2)
        print(f"[Save] val_oof_seed{seed}.npy: {val_oof.shape}, NaN frac={np.isnan(val_oof).mean():.3f}", flush=True)

        # test OOF
        test_dates_arr = test_meta['dates']
        test_date_to_idx = {d: i for i, d in enumerate(test_dates_arr)}
        test_oof = np.full((len(test_dates_arr), len(all_stocks_full)), np.nan, dtype=np.float32)
        test_proc = processed.loc[test_mask].copy()
        test_proc['_d_str'] = test_proc['日期'].dt.strftime('%Y-%m-%d')
        test_proc['_d_idx'] = test_proc['_d_str'].map(test_date_to_idx)
        test_proc['_s_idx'] = test_proc['股票代码'].map(stockid2idx)
        test_valid = test_proc['_d_idx'].notna() & test_proc['_s_idx'].notna()
        test_oof[test_proc.loc[test_valid, '_d_idx'].astype(int).values,
                 test_proc.loc[test_valid, '_s_idx'].astype(int).values] = test_pred_all[test_valid.values]
        np.save(os.path.join(args.output_dir, f'oof_scores_seed{seed}.npy'), test_oof)
        with open(os.path.join(args.output_dir, 'test_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(test_meta, f, ensure_ascii=False, indent=2)
        print(f"[Save] oof_scores_seed{seed}.npy (test): {test_oof.shape}, NaN frac={np.isnan(test_oof).mean():.3f}", flush=True)


if __name__ == '__main__':
    main()
