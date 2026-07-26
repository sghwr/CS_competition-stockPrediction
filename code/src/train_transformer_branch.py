"""MacroTransformer branch.

Train MacroIndustryAlphaBetaModel on market data (HS300 + 11 industries x 3 features), predict:
  - Per-day per-industry alpha (B, 11)
  - Per-day per-industry beta (B, 11)

设计:
  1. 3 维特征: open_norm, vol_20d, open_mom_5d (log_return based)
  2. Pre-LN Transformer + cross-industry attention
  3. Industry embedding
  4. 0.5 Huber(alpha) + 0.5 Huber(beta) loss
  5. Multi-seed bagging (默认 3 seeds, 平均输出)
  6. regime 用确定性规则 (regime_classifier), 不训

No individual stock prediction.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# F imported above for cross_entropy in regime_8 loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from model import MacroTransformerV4, build_macro_features
from paths import INTEGRATED_DIR, MODEL_INTEGRATED_DIR, TRAIN_CSV, DATA_DIR, INDEX_CSV
from industry_mapping import load_industry_map


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def focal_loss(logits, targets, alpha=None, gamma=2.0):
    """Focal loss for class imbalance (Lin et al. 2017).
    logits: (B, C), targets: (B,)
    """
    ce = F.cross_entropy(logits, targets, reduction='none')  # (B,)
    pt = torch.exp(-ce)  # prob of correct class
    focal = ((1 - pt) ** gamma) * ce
    if alpha is not None:
        alpha_t = alpha[targets]
        focal = alpha_t * focal
    return focal.mean()


def build_targets(macro_dates, df, industry_map, beta_lookback=60):
    """Build training targets for MacroIndustryAlphaBetaModel.

    只训 industry alpha + beta, 简化任务.
    regime 用确定性规则 (regime_classifier), 不训.

    Targets per day:
      - industry_alpha: 11-dim, mean of (open[t+5]-open[t+1])/open[t+1] per industry
                        - HS300 5d open return (industry rotation signal)
      - industry_beta: 11-dim, rolling beta (corr) of industry daily return
                       vs HS300 daily return over past `beta_lookback` days
                       范围 [-1, 1], 高 beta = 市场敏感 (银行/地产), 低 beta = 防御 (医药)

    Returns:
      industry_alpha: (T, 11) float32
      industry_beta:  (T, 11) float32
    """
    import pandas as pd
    idx = pd.read_csv(INDEX_CSV, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    idx['date_str'] = idx['date'].dt.strftime('%Y-%m-%d')
    idx['ret_5d'] = ((idx['open'].shift(-5) - idx['open'].shift(-1))
                     / (idx['open'].shift(-1) + 1e-12)).fillna(0)
    hs300_5d = dict(zip(idx['date_str'], idx['ret_5d']))

    industries = sorted(set(industry_map.values()))
    n_ind = len(industries)
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}

    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df['industry_idx'] = df['股票代码'].map(
        lambda c: industry_to_idx.get(industry_map.get(int(c), ''), 0)
    )
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # Per stock 5d forward return (open t+1 to open t+5)
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['ret_5d'] = ((df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)).clip(-0.5, 0.5)

    # Per day per industry: mean stock 5d return
    ind_5d = df.groupby(['日期', 'industry_idx'])['ret_5d'].mean().reset_index(name='_ind_5d')

    T = len(macro_dates)
    hs300_5d_arr = np.array([hs300_5d.get(d, 0) for d in macro_dates], dtype=np.float32)

    # industry_alpha = ind_5d_mean - hs300_5d (excess return)
    ind_pivot = ind_5d.pivot_table(
        index='日期', columns='industry_idx', values='_ind_5d', aggfunc='first',
    )
    ind_mat = np.zeros((T, n_ind), dtype=np.float32)
    for ii in range(n_ind):
        if ii in ind_pivot.columns:
            series = ind_pivot[ii].reindex(macro_dates).fillna(0).values
            ind_mat[:, ii] = series
    industry_alpha = ind_mat - hs300_5d_arr[:, None].astype(np.float32)

    # industry_beta: 滚动 60d 相关性 vs HS300
    print(f"[Target] building industry_beta (rolling {beta_lookback}d corr) ...", flush=True)
    ind_daily_open = df.groupby(['日期', 'industry_idx'])['开盘'].mean().reset_index(name='_open')
    ind_daily_open = ind_daily_open.sort_values(['industry_idx', '日期']).reset_index(drop=True)
    idx['log_ret'] = np.log(idx['close'] / idx['close'].shift(1)).fillna(0)
    hs300_daily_ret = dict(zip(idx['date_str'], idx['log_ret']))

    industry_beta_arr = np.zeros((T, n_ind), dtype=np.float32)
    for ii in range(n_ind):
        sub = ind_daily_open[ind_daily_open['industry_idx'] == ii].copy()
        if len(sub) < beta_lookback + 5:
            continue
        sub = sub.sort_values('日期').reset_index(drop=True)
        sub['log_ret'] = np.log(sub['_open'] / sub['_open'].shift(1)).fillna(0)
        sub['date_str'] = sub['日期'].dt.strftime('%Y-%m-%d')
        sub['hs300_ret'] = sub['date_str'].map(hs300_daily_ret).fillna(0)
        sub['rolling_beta'] = sub['log_ret'].rolling(beta_lookback, min_periods=20).corr(
            sub['hs300_ret']
        ).fillna(0).clip(-1, 1)
        beta_dict = dict(zip(sub['date_str'], sub['rolling_beta']))
        for t, d_str in enumerate(macro_dates):
            industry_beta_arr[t, ii] = beta_dict.get(d_str, 0)
    print(f"[Target] industry_alpha: mean={industry_alpha.mean():.4f}, "
          f"std={industry_alpha.std():.4f}, "
          f"range=[{industry_alpha.min():.4f}, {industry_alpha.max():.4f}]", flush=True)
    print(f"[Target] industry_beta: mean={industry_beta_arr.mean():.3f}, "
          f"std={industry_beta_arr.std():.3f}, "
          f"range=[{industry_beta_arr.min():.3f}, {industry_beta_arr.max():.3f}]", flush=True)
    return industry_alpha, industry_beta_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7],
                        help='默认 3 seeds, 平均输出')
    parser.add_argument('--model_dir', type=str, default=str(MODEL_INTEGRATED_DIR / 'macro_transformer'),
                        help='模型权重目录 (best_seed*.pth)')
    parser.add_argument('--output_dir', type=str, default=str(INTEGRATED_DIR / 'macro_transformer'),
                        help='推理产物目录 (predictions, meta)')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n{'=' * 80}\n>>> [MacroIndustryAlphaBetaModel] starting\n", flush=True)
    print(f"[Config] max_epochs={args.epochs}, patience={args.patience}, seeds={args.seeds}, "
          f"lr={args.lr}, smoke={args.smoke}", flush=True)
    print(f"[Config] val={config['val_start']} ~ {config['val_end']} (from config.SPLITS)", flush=True)
    print(f"[Config] model_dir = {args.model_dir}", flush=True)
    print(f"[Config] output_dir = {args.output_dir}", flush=True)

    # 1. Load stock data
    print("\n[Data] loading stock_data.csv ...", flush=True)
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    print(f"[Data] {len(df):,} rows, {df['股票代码'].nunique()} stocks, "
          f"{df['日期'].min().date()} ~ {df['日期'].max().date()}", flush=True)

    if args.smoke:
        sample_stocks = sorted(df['股票代码'].unique())[:30]
        df = df[df['股票代码'].isin(sample_stocks)].copy()
        all_dates = sorted(df['日期'].unique())
        df = df[df['日期'].isin(all_dates[-500:])].copy()
        print(f"[Smoke] {len(df):,} rows", flush=True)

    # 2. Industry mapping
    print("\n[Industry] loading industry map ...", flush=True)
    industry_map = load_industry_map()

    # 3. Build macro features (3 维 log_return, 删 log_volume/RSI)
    print("\n[Macro] building HS300 + 11 industries x 60d x 3 features (log_return) ...", flush=True)
    t0 = time.time()
    macro_X, macro_dates, industries = build_macro_features(
        df, industry_map, INDEX_CSV, seq_len=60,
    )
    n_features = macro_X.shape[-1]
    print(f"[Macro] {macro_X.shape}, {len(macro_dates)} dates, "
          f"{len(industries)} industries, {n_features} features, in {time.time()-t0:.1f}s", flush=True)

    # 4. Build targets (only industry_alpha + industry_beta, no regime)
    print("\n[Target] building industry_alpha + industry_beta ...", flush=True)
    t0 = time.time()
    industry_alpha, industry_beta = build_targets(
        macro_dates, df, industry_map,
    )
    print(f"[Target] done in {time.time()-t0:.1f}s", flush=True)

    # 5. Train/val/test split by date (3 段)
    val_start = pd.Timestamp(config['val_start'])
    val_end = pd.Timestamp(config['val_end'])
    test_start = pd.Timestamp(config['test_start'])
    test_end = pd.Timestamp(config['test_end'])
    train_mask = np.array([pd.Timestamp(d) < val_start for d in macro_dates])
    val_mask = np.array([(pd.Timestamp(d) >= val_start) & (pd.Timestamp(d) <= val_end)
                         for d in macro_dates])
    test_mask = np.array([(pd.Timestamp(d) >= test_start) & (pd.Timestamp(d) <= test_end)
                          for d in macro_dates])
    print(f"[Split] train: {train_mask.sum()}, val: {val_mask.sum()}, "
          f"test: {test_mask.sum()}", flush=True)

    # 6. Tensors
    X_all = torch.from_numpy(macro_X).float()
    ind_alpha_t = torch.from_numpy(industry_alpha).float()
    ind_beta_t = torch.from_numpy(industry_beta).float()

    # 7. Multi-seed training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}", flush=True)

    all_seed_preds = []
    for seed in args.seeds:
        print(f"\n========== Seed {seed} ==========", flush=True)
        set_seed(seed)

        model = MacroTransformerV4(
            n_channels=12, n_features=n_features, seq_len=60,
            d_model=128, nhead=4, num_layers=3,
            n_industries=len(industries), dropout=0.2,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        from torch.optim.lr_scheduler import LambdaLR
        warmup_epochs = 3
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)))
        scheduler = LambdaLR(optimizer, lr_lambda)

        huber_loss = nn.HuberLoss(delta=0.1)

        best_val_loss = float('inf')
        best_state = None
        no_improve = 0

        train_X = X_all[train_mask].to(device)
        train_alpha = ind_alpha_t[train_mask].to(device)
        train_beta = ind_beta_t[train_mask].to(device)
        val_X = X_all[val_mask].to(device)
        val_alpha = ind_alpha_t[val_mask].to(device)
        val_beta = ind_beta_t[val_mask].to(device)

        for epoch in range(args.epochs):
            t0 = time.time()
            model.train()
            B = 32
            perm = torch.randperm(len(train_X))
            tr_loss = 0.0
            n_batch = 0
            for i in range(0, len(perm), B):
                idx = perm[i:i + B]
                xb = train_X[idx]
                ind_alpha_pred, ind_beta_pred = model(xb)
                l_alpha = huber_loss(ind_alpha_pred, train_alpha[idx])
                l_beta = huber_loss(ind_beta_pred, train_beta[idx])
                # pairwise ranking loss for alpha: 鼓励行业排序正确
                pred_diff = ind_alpha_pred.unsqueeze(2) - ind_alpha_pred.unsqueeze(1)
                true_diff = train_alpha[idx].unsqueeze(2) - train_alpha[idx].unsqueeze(1)
                rank_loss = torch.mean(torch.clamp(-true_diff * pred_diff + 0.01, min=0))
                loss = 0.4 * l_alpha + 0.4 * l_beta + 0.2 * rank_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item()
                n_batch += 1
            tr_loss /= max(n_batch, 1)

            scheduler.step()

            model.eval()
            with torch.no_grad():
                ind_alpha_pred, ind_beta_pred = model(val_X)
                l_alpha_v = huber_loss(ind_alpha_pred, val_alpha).item()
                l_beta_v = huber_loss(ind_beta_pred, val_beta).item()
                val_loss = 0.5 * l_alpha_v + 0.5 * l_beta_v

            print(f"[seed={seed} epoch={epoch+1}/{args.epochs}] "
                  f"tr={tr_loss:.4f} | val alpha={l_alpha_v:.4f} beta={l_beta_v:.4f} "
                  f"| {time.time()-t0:.1f}s", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                torch.save(best_state, os.path.join(args.model_dir, f'best_seed{seed}.pth'))
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"[seed={seed}] early stop at epoch={epoch+1}, best={best_val_loss:.4f}", flush=True)
                    break

        # Inference
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            ind_alpha_pred, ind_beta_pred = model(X_all.to(device))
        ind_alpha_all = ind_alpha_pred.cpu().numpy()
        ind_beta_all = ind_beta_pred.cpu().numpy()

        all_seed_preds.append({
            'dates': macro_dates,
            'industry_bias': ind_alpha_all,
            'industry_beta': ind_beta_all,
            'industries': industries,
        })

        np.savez(os.path.join(args.output_dir, f'macro_pred_seed{seed}.npz'),
                 dates=macro_dates,
                 industry_bias=ind_alpha_all,
                 industry_beta=ind_beta_all,
                 industries=np.array(industries, dtype=object))
        print(f"[Save] macro_pred_seed{seed}.npz: "
              f"alpha=[{ind_alpha_all.min():.3f}, {ind_alpha_all.max():.3f}], "
              f"beta=[{ind_beta_all.min():.3f}, {ind_beta_all.max():.3f}]",
              flush=True)

    # 8. Multi-seed average (no regime_pred, regime_8_prob)
    if len(args.seeds) > 1:
        print(f"\n[Bagging] averaging {len(args.seeds)} seeds ...", flush=True)
        industry_bias_avg = np.mean([p['industry_bias'] for p in all_seed_preds], axis=0)
        industry_beta_avg = np.mean([p['industry_beta'] for p in all_seed_preds], axis=0)
        np.savez(os.path.join(args.output_dir, 'macro_pred_avg.npz'),
                 dates=macro_dates,
                 industry_bias=industry_bias_avg,
                 industry_beta=industry_beta_avg,
                 industries=np.array(industries, dtype=object))
        print(f"[Save] macro_pred_avg.npz (avg of {len(args.seeds)} seeds): "
              f"alpha=[{industry_bias_avg.min():.3f}, {industry_bias_avg.max():.3f}], "
              f"beta=[{industry_beta_avg.min():.3f}, {industry_beta_avg.max():.3f}]",
              flush=True)

    # 9. Save val_meta.json (val 段 9 月, 早停 + stack 训练范围) + test_meta.json (test 段 3 月, 评估范围)
    val_dates_only = [d for d in macro_dates
                      if val_start <= pd.Timestamp(d) <= val_end]
    val_meta = {
        'dates': val_dates_only,
        'stock_indices': [list(range(300)) for _ in val_dates_only],
        'n_stocks': 300,
        'seq_len': 60,
    }
    with open(os.path.join(args.output_dir, 'val_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(val_meta, f, ensure_ascii=False, indent=2)
    print(f"[Save] val_meta.json: {len(val_dates_only)} val dates "
          f"({val_dates_only[0]} ~ {val_dates_only[-1]})", flush=True)

    test_dates_only = [d for d in macro_dates
                       if test_start <= pd.Timestamp(d) <= test_end]
    test_meta = {
        'dates': test_dates_only,
        'stock_indices': [list(range(300)) for _ in test_dates_only],
        'n_stocks': 300,
        'seq_len': 60,
    }
    with open(os.path.join(args.output_dir, 'test_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(test_meta, f, ensure_ascii=False, indent=2)
    print(f"[Save] test_meta.json: {len(test_dates_only)} test dates "
          f"({test_dates_only[0]} ~ {test_dates_only[-1]})", flush=True)


if __name__ == '__main__':
    main()
