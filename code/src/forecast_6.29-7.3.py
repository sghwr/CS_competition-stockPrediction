"""Forecast for 2026-06-29 ~ 2026-07-03 (5 trading days).

Inference pipeline:
  1. MacroTransformer: 训 in train, 推理 6.29-7.3 → industry_bias + industry_beta
  2. Tree (3 regime-specific): 训 in train (per-regime), 推理 6.29-7.3 → tree_pred per regime
  3. Linear: 训 in train, 推理 6.29-7.3 → linear_pred
  4. Stack (3 regime-specific): 训 in val, 推理 6.29-7.3 → final_pred (按 regime_pred 选 model)
  5. 输出 result.csv: top-5 picks + equal weight 0.2
     selection: bear/bull=top5_ic (alpha 自由), sideways=industry_diversified (避免行业集中)

NO LEAK:
  - 训 in train (2015-2025-06), 推理 6.29-7.3 (2026)
  - Features 全用 past data, 不含 future
  - 推理不接触 labels

Output format:
  stock_id, weight
  - 最多 5 个不同的 stock_id
  - weight 累加 ≤ 1
  - 不到 1 持有现金
"""
import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import INTEGRATED_DIR, MODEL_INTEGRATED_DIR, TRAIN_CSV, INDEX_CSV
from config import config
from model import (MacroIndustryAlphaBetaModel, LinearRegressionModel, build_macro_features)
from industry_mapping import load_industry_map

FORECAST_DATES = ['2026-06-29', '2026-06-30', '2026-07-01', '2026-07-02', '2026-07-03']


def get_test_dates_forecast():
    """Forecast dates: 5 trading days after test set."""
    return FORECAST_DATES


def zscore_per_day(oof):
    """Per-day z-score normalization. NaN rows left as NaN."""
    z = oof.copy()
    for d in range(z.shape[0]):
        row = z[d]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        m = row[valid].mean()
        s = row[valid].std()
        if s > 1e-9:
            z[d, valid] = (row[valid] - m) / s
        else:
            z[d, valid] = 0
    return z


def run_macro_inference(forecast_dates, val_meta, device):
    """MacroIndustryAlphaBetaModel: 对 forecast 期间 inference.

    策略: 用 train+val+test+forecast 数据 build_macro_features,
    取 forecast 日期对应的窗口. 对每个 seed 模型 inference, 平均输出.
    """
    print("\n[MacroTransformer] loading model ...", flush=True)
    macro_model_dir = MODEL_INTEGRATED_DIR / 'macro_transformer'
    seed_files = sorted(macro_model_dir.glob('best_seed*.pth'))
    if not seed_files:
        raise FileNotFoundError(f"No best_seed*.pth in {macro_model_dir}")
    print(f"  found {len(seed_files)} seeds: {[f.name for f in seed_files]}", flush=True)

    # 加载所有数据 + build macro features (3 features, log_return)
    print("[MacroTransformer] loading data + building macro features (3 dim, log_return) ...", flush=True)
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    industry_map = load_industry_map()
    macro_X, macro_dates_all, industries = build_macro_features(
        df, industry_map, INDEX_CSV, seq_len=60,
    )
    print(f"  macro_X: {macro_X.shape}, dates: {len(macro_dates_all)}", flush=True)

    # 取 forecast 日期对应的索引
    forecast_set = set(forecast_dates)
    forecast_indices = [i for i, d in enumerate(macro_dates_all) if d in forecast_set]
    if not forecast_indices:
        raise ValueError(f"No forecast dates found in macro_dates_all (last: {macro_dates_all[-1]})")
    forecast_macro_X = macro_X[forecast_indices]
    forecast_macro_dates = [macro_dates_all[i] for i in forecast_indices]
    print(f"  forecast_macro_X: {forecast_macro_X.shape}, dates: {len(forecast_macro_dates)}", flush=True)

    n_industries = len(industries)
    X_t = torch.from_numpy(forecast_macro_X).float().to(device)

    all_preds = []
    for sf in seed_files:
        model = MacroIndustryAlphaBetaModel(
            n_channels=12, n_features=3, seq_len=60,
            d_model=128, nhead=4, num_layers=3,
            n_industries=n_industries, dropout=0.2,
        ).to(device)
        state = torch.load(sf, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            ind_alpha, ind_beta = model(X_t)
        all_preds.append({
            'industry_bias': ind_alpha.cpu().numpy(),
            'industry_beta': ind_beta.cpu().numpy(),
        })
        del model

    avg = {
        'dates': forecast_macro_dates,
        'industry_bias': np.mean([p['industry_bias'] for p in all_preds], axis=0),
        'industry_beta': np.mean([p['industry_beta'] for p in all_preds], axis=0),
        'industries': industries,
    }
    print(f"[MacroTransformer] forecast inference done: industry_alpha range=[{avg['industry_bias'].min():.3f}, "
          f"{avg['industry_bias'].max():.3f}], industry_beta range=[{avg['industry_beta'].min():.3f}, "
          f"{avg['industry_beta'].max():.3f}]", flush=True)
    return avg


def run_tree_inference_per_regime(forecast_dates, val_meta):
    """Tree (3 regime-specific): 对 forecast 期间 inference.

    加载 3 个 model_{bear,sideways,bull}.txt, 算 187 dim features, inference.
    """
    print("\n[Tree: 3 regime-specific] loading models ...", flush=True)
    import lightgbm as lgb
    from utils import engineer_features_158plus39, TIER1_LAST_ONLY, TIER2_ENRICHED, \
        add_industry_beta_features
    from tree_branch import add_industry_features
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    tree_dir = INTEGRATED_DIR / 'tree'  # artifacts (scaler, feature_names)
    tree_model_dir = MODEL_INTEGRATED_DIR / 'tree'  # model weights
    feature_names = joblib.load(tree_dir / 'feature_names.pkl')
    scaler = joblib.load(tree_dir / 'scaler.pkl')

    # 加载 3 个 model
    models = {}
    for r_name in ['bear', 'sideways', 'bull']:
        model_p = tree_model_dir / f'model_{r_name}.txt'
        if model_p.exists():
            models[r_name] = lgb.Booster(model_file=str(model_p))
            print(f"  loaded model_{r_name}.txt", flush=True)

    # Load data + filter forecast
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    forecast_set = set(forecast_dates)
    df['_d_str'] = df['日期'].dt.strftime('%Y-%m-%d')
    df_forecast = df[df['_d_str'].isin(forecast_set)].copy()
    print(f"  forecast rows: {len(df_forecast):,}", flush=True)

    # Engineer features (需要历史 context: T-59 ~ T)
    print("[Tree] engineering features ...", flush=True)
    val_end_dt = pd.Timestamp(config['val_end'])
    context_start = val_end_dt - pd.tseries.offsets.BDay(120)
    df_full = df[(df['日期'] >= context_start)].copy()
    print(f"  context+forecast: {len(df_full):,} rows", flush=True)

    all_groups = [g for _, g in df_full.groupby('股票代码', sort=False)]
    n_thread = min(8, (os.cpu_count() or 1) * 2)
    processed_list = [None] * len(all_groups)
    with ThreadPoolExecutor(max_workers=n_thread) as ex:
        futures = {ex.submit(engineer_features_158plus39, g): i for i, g in enumerate(all_groups)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="engineer", mininterval=2.0):
            processed_list[futures[fut]] = fut.result()
    processed = pd.concat(processed_list).reset_index(drop=True)
    print(f"  engineered: {len(processed):,} rows", flush=True)

    # Clean
    cols_to_clean = TIER1_LAST_ONLY + TIER2_ENRICHED
    cols_present = [c for c in cols_to_clean if c in processed.columns]
    for c in cols_present:
        processed[c] = processed[c].replace([np.inf, -np.inf], 0).fillna(0)

    # 行业特征
    industry_map = load_industry_map()
    processed = add_industry_features(processed, industry_map)
    print(f"  cols: {len(processed.columns)}", flush=True)

    # 过滤 forecast 期, 取 T-59..T 窗口末值
    forecast_set_str = set(forecast_dates)
    proc_groups = {}
    for sid, g in processed.groupby('股票代码', sort=False):
        g = g.sort_values('日期').reset_index(drop=True)
        g['_d_str'] = g['日期'].dt.strftime('%Y-%m-%d')
        for i in range(60, len(g)):
            d_str = g.iloc[i]['_d_str']
            if d_str in forecast_set_str:
                window = g.iloc[i - 60:i]
                proc_groups[d_str][sid] = window.iloc[-1][feature_names].values if d_str in proc_groups else {sid: window.iloc[-1][feature_names].values}

    # Predict per regime
    print(f"[Tree] predicting on {len(forecast_dates)} forecast dates ...", flush=True)
    all_stocks = sorted(df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_stocks)}

    # 按 forecast date 顺序
    forecast_ordered = [d for d in forecast_dates if d in proc_groups]

    # 为每个 regime 跑一遍, 输出 (n_forecast, n_stocks)
    regime_oofs = {}
    for r_name, lgbm in models.items():
        oof = np.full((len(forecast_dates), len(all_stocks)), np.nan, dtype=np.float32)
        for d_str in forecast_ordered:
            if d_str not in proc_groups:
                continue
            d_idx = forecast_dates.index(d_str)
            rows = []
            sids = []
            for sid, feat in proc_groups[d_str].items():
                feat = np.asarray(feat, dtype=np.float64)
                if not np.any(np.isnan(feat)):
                    rows.append(feat)
                    sids.append(sid)
            if not rows:
                continue
            X = np.array(rows, dtype=np.float32)
            X_s = scaler.transform(X)
            pred = lgbm.predict(X_s)
            for sid, p in zip(sids, pred):
                oof[d_idx, stockid2idx[sid]] = p
        regime_oofs[r_name] = oof
        print(f"  Tree_{r_name} forecast: shape={oof.shape}, NaN={np.isnan(oof).mean():.3f}", flush=True)
    return regime_oofs


def run_linear_inference(forecast_dates, val_meta):
    """LinearRegression: 对 forecast 期间 inference (10 dim reversal features)."""
    print("\n[LinearRegression: 10 dim reversal] loading model ...", flush=True)
    linear_dir = INTEGRATED_DIR / 'linear_regression'  # artifacts
    linear_model_dir = MODEL_INTEGRATED_DIR / 'linear_regression'  # weights
    seed_files = sorted(linear_model_dir.glob('best_seed*.pth'))
    if not seed_files:
        raise FileNotFoundError(f"No best_seed*.pth in {linear_model_dir}")
    print(f"  found {len(seed_files)} seeds: {[f.name for f in seed_files]}", flush=True)

    scaler = joblib.load(linear_dir / 'scaler.pkl')
    feature_cols = joblib.load(linear_dir / 'feature_names.pkl')
    print(f"  features: {len(feature_cols)} ({feature_cols})", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from mdsrp_regression import build_reversal_features, REVERSAL_FEATURES
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    forecast_set = set(forecast_dates)
    df['_d_str'] = df['日期'].dt.strftime('%Y-%m-%d')

    # 加载 forecast 期数据 + 必要 context
    val_end_dt = pd.Timestamp(config['val_end'])
    context_start = val_end_dt - pd.tseries.offsets.BDay(120)
    df_full = df[(df['日期'] >= context_start)].copy()

    print("[Linear] building 10 reversal features (NO macro, past only) ...", flush=True)
    processed = build_reversal_features(df_full)
    for c in feature_cols:
        if c in processed.columns and c not in REVERSAL_FEATURES:
            processed[c] = processed[c].replace([np.inf, -np.inf], 0).fillna(0)

    # Filter forecast
    forecast_set_str = set(forecast_dates)
    proc_groups = {}
    for sid, g in processed.groupby('股票代码', sort=False):
        g = g.sort_values('日期').reset_index(drop=True)
        g['_d_str'] = g['日期'].dt.strftime('%Y-%m-%d')
        for i in range(20, len(g)):
            d_str = g.iloc[i]['_d_str']
            if d_str in forecast_set_str:
                window = g.iloc[i - 20:i]
                if d_str not in proc_groups:
                    proc_groups[d_str] = {}
                proc_groups[d_str][sid] = window.iloc[-1][feature_cols].values

    print(f"[Linear] predicting on {len(forecast_dates)} forecast dates ...", flush=True)
    all_stocks = sorted(df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_stocks)}

    # Average across seeds
    oof_avg = np.zeros((len(forecast_dates), len(all_stocks)), dtype=np.float32)
    for sf in seed_files:
        model = LinearRegressionModel(n_features=len(feature_cols), output_clip=0.3).to(device)
        state = torch.load(sf, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        for d_str in forecast_dates:
            if d_str not in proc_groups:
                continue
            d_idx = forecast_dates.index(d_str)
            rows = []
            sids = []
            for sid, feat in proc_groups[d_str].items():
                feat = np.asarray(feat, dtype=np.float64)
                if not np.any(np.isnan(feat)):
                    rows.append(feat)
                    sids.append(sid)
            if not rows:
                continue
            X = np.array(rows, dtype=np.float32)
            X_s = scaler.transform(X)
            with torch.no_grad():
                X_t = torch.from_numpy(X_s).float().to(device)
                pred = model(X_t).cpu().numpy()
            for sid, p in zip(sids, pred):
                oof_avg[d_idx, stockid2idx[sid]] += p
        del model
    oof_avg /= len(seed_files)
    print(f"[Linear] forecast: shape={oof_avg.shape}, NaN={np.isnan(oof_avg).mean():.3f}", flush=True)
    return oof_avg


def build_signals_for_forecast(forecast_dates, all_stocks):
    """Build 5 explicit signals (reversal/momentum/beta) for forecast dates.

    NO LEAK: past data only.
    """
    print("\n[Signals] building 5 explicit signals for forecast (NO LEAK) ...", flush=True)
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # past_ret_5d (no future)
    df['past_ret_5d'] = (df['收盘'] / df.groupby('股票代码')['收盘'].shift(5) - 1).clip(-0.3, 0.3)

    # 60d rolling beta vs HS300
    idx = pd.read_csv(INDEX_CSV, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    idx['_d_str'] = idx['date'].dt.strftime('%Y-%m-%d')
    idx['log_ret'] = np.log(idx['close'] / idx['close'].shift(1)).fillna(0)
    hs300_ret_map = dict(zip(idx['_d_str'], idx['log_ret']))

    df['log_ret'] = np.log(df['收盘'] / df['收盘'].shift(1)).fillna(0)
    df['hs300_ret'] = df['日期'].dt.strftime('%Y-%m-%d').map(hs300_ret_map).fillna(0)
    df['_d_str'] = df['日期'].dt.strftime('%Y-%m-%d')

    print("[Signals] computing 60d rolling beta vs HS300 ...", flush=True)
    df['beta_60'] = (df.groupby('股票代码', sort=False)
                      .apply(lambda g: g['log_ret'].rolling(60, min_periods=20).corr(g['hs300_ret']),
                             include_groups=False)
                      .reset_index(level=0, drop=True))

    date_to_idx = {d: i for i, d in enumerate(forecast_dates)}
    stock_to_idx = {int(s): i for i, s in enumerate(all_stocks)}
    n_days, n_stocks = len(forecast_dates), len(all_stocks)

    def pivot(col):
        mat = np.full((n_days, n_stocks), 0.0, dtype=np.float32)
        sub = df[['日期', '股票代码', col]].copy()
        sub['_d_str'] = sub['日期'].dt.strftime('%Y-%m-%d')
        for _, row in sub.iterrows():
            d = row['_d_str']
            s = int(row['股票代码'])
            if d in date_to_idx and s in stock_to_idx:
                v = row[col]
                if pd.notna(v):
                    mat[date_to_idx[d], stock_to_idx[s]] = float(v)
        return mat

    past_ret_5d = pivot('past_ret_5d')
    beta_60 = pivot('beta_60')

    return {
        'reversal_score': -past_ret_5d,
        'momentum_score': past_ret_5d,
        'low_beta_score': -beta_60.clip(-1, 1),
        'high_beta_score': beta_60.clip(-1, 1),
        'diversified_score': np.zeros((n_days, n_stocks), dtype=np.float32),
        'macro_industry_beta': beta_60,
    }


def get_regime_for_dates(dates):
    """Get 3-class regime (0=bear, 1=sideways, 2=bull) for each date."""
    from regime_classifier import RegimeClassifier
    rc = RegimeClassifier()
    regime_pred, _ = rc.predict(dates)
    return regime_pred


def stack_inference(regime_models, regime_oofs, z_lin, test_signals, test_regime,
                     test_macro, industry_map, all_stocks, dates, n_features_per_regime):
    """Run 3 regime stack models on test data, route by regime_pred."""
    n_days = len(dates)
    n_stocks = len(all_stocks)
    final = np.zeros((n_days, n_stocks), dtype=np.float32)

    industries = sorted(set(industry_map.values()))
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    stock_industry = np.array([
        industry_to_idx.get(industry_map.get(int(s), ''), 0)
        for s in all_stocks
    ])

    macro_date_to_idx = {d: i for i, d in enumerate(test_macro['dates'])}
    macro_industries = test_macro['industries']
    macro_ind_to_idx = {ind: i for i, ind in enumerate(macro_industries)}
    full_to_macro = np.array([macro_ind_to_idx.get(ind, -1) for ind in industries], dtype=np.int32)
    valid_macro = full_to_macro >= 0

    # Pre-compute macro arrays
    macro_bias_mat = np.zeros((n_days, len(industries)), dtype=np.float32)
    macro_beta_mat = np.zeros((n_days, len(industries)), dtype=np.float32)
    for d_idx, d_str in enumerate(dates):
        m_idx = macro_date_to_idx.get(d_str)
        if m_idx is not None and 'industry_bias' in test_macro:
            macro_bias = test_macro['industry_bias'][m_idx]
            for i in range(len(industries)):
                if valid_macro[i]:
                    macro_bias_mat[d_idx, i] = macro_bias[full_to_macro[i]]
        if m_idx is not None and 'industry_beta' in test_macro:
            macro_beta = test_macro['industry_beta'][m_idx]
            for i in range(len(industries)):
                if valid_macro[i]:
                    macro_beta_mat[d_idx, i] = macro_beta[full_to_macro[i]]

    # Build X for each day, route to regime model
    REGIME_NAMES = ['bear', 'sideways', 'bull']
    REGIME_STRATEGY_FEATURES = {
        0: ['reversal_score', 'low_beta_score'],
        1: ['macro_industry_beta', 'diversified_score'],
        2: ['momentum_score', 'high_beta_score'],
    }
    COMMON_BASE_FEATURES = ['tree_pred', 'linear_pred', 'macro_industry_bias', 'industry_id']

    for d_idx, d_str in enumerate(dates):
        r = int(test_regime[d_idx])
        r_name = REGIME_NAMES[r]
        model_r = regime_models.get(r_name)
        if model_r is None:
            model_r = regime_models.get('sideways') or regime_models.get('bear') or regime_models.get('bull')
            r = 1  # fallback to sideways features

        n_features = n_features_per_regime[r]
        X_day = np.zeros((n_stocks, n_features), dtype=np.float32)
        tree_oof = regime_oofs.get(r_name, regime_oofs.get('sideways'))
        for s_idx in range(n_stocks):
            sid = stock_industry[s_idx]
            col = 0
            t_pred = tree_oof[d_idx, s_idx] if not np.isnan(tree_oof[d_idx, s_idx]) else 0
            l_pred = z_lin[d_idx, s_idx] if not np.isnan(z_lin[d_idx, s_idx]) else 0
            X_day[s_idx, col] = t_pred; col += 1
            X_day[s_idx, col] = l_pred; col += 1
            X_day[s_idx, col] = macro_bias_mat[d_idx, sid]; col += 1
            X_day[s_idx, col] = sid; col += 1
            # regime-specific signals
            for sig_name in REGIME_STRATEGY_FEATURES[r]:
                sig_mat = test_signals[sig_name]
                val_ = sig_mat[d_idx, s_idx] if d_idx < sig_mat.shape[0] else 0
                X_day[s_idx, col] = val_; col += 1

        pred = model_r.predict(X_day)
        final[d_idx] = pred
    return final


def main():
    base_dir = INTEGRATED_DIR
    forecast_dates = get_test_dates_forecast()
    print(f"[Forecast] dates: {forecast_dates}", flush=True)
    print(f"[Strategy] 6.27-6.28 是周末没数据, 用 2026-06-26 (test 段最后 1 天) 单天 OOF 作为 forecast proxy", flush=True)

    all_stocks = sorted(pd.read_csv(TRAIN_CSV, encoding='utf-8-sig',
                                     usecols=['股票代码'])['股票代码'].unique())
    industry_map = load_industry_map()

    # 1. MacroTransformer: 用 2026-06-26 单天 industry_bias + industry_beta
    print("\n[MacroTransformer] using 2026-06-26 single day ...", flush=True)
    macro_p = base_dir / 'macro_transformer' / 'macro_pred_avg.npz'
    macro_npz = np.load(macro_p, allow_pickle=True)
    macro_all = {
        'dates': list(macro_npz['dates']),
        'industry_bias': macro_npz['industry_bias'],
        'industry_beta': macro_npz['industry_beta'],
        'industries': list(macro_npz['industries']),
    }
    # 2026-06-26 单天 (test 段最后 1 天)
    last_day = '2026-06-26'
    if last_day not in macro_all['dates']:
        last_day = macro_all['dates'][-1]
    last_idx = macro_all['dates'].index(last_day)
    last_bias = macro_all['industry_bias'][last_idx]  # (11,)
    last_beta = macro_all['industry_beta'][last_idx]
    # test_meta_idx: 在 test 段里的 index (0-113)
    test_meta = json.load(open(base_dir / 'macro_transformer' / 'test_meta.json', encoding='utf-8'))
    test_last_idx = test_meta['dates'].index(last_day)
    test_macro = {
        'dates': forecast_dates,
        'industry_bias': np.tile(last_bias[None, :], (len(forecast_dates), 1)),
        'industry_beta': np.tile(last_beta[None, :], (len(forecast_dates), 1)),
        'industries': macro_all['industries'],
    }
    print(f"  Using {last_day}: bias range=[{last_bias.min():.3f}, {last_bias.max():.3f}]")

    # 2. Tree (3 regime): 用 2026-06-26 单天 OOF
    print("\n[Tree: 3 regime] using 2026-06-26 single day ...", flush=True)
    z_regime_oofs = {}
    for r_name in ['bear', 'sideways', 'bull']:
        test_p = base_dir / 'tree' / f'oof_scores_{r_name}.npy'
        if test_p.exists():
            oof = np.load(test_p)
            z_oof = zscore_per_day(oof[test_last_idx:test_last_idx+1])[0]  # z-score 1 天
            z_regime_oofs[r_name] = np.tile(z_oof[None, :], (len(forecast_dates), 1))
            print(f"  Tree_{r_name} (2026-06-26) z-score range=[{z_oof.min():.3f}, {z_oof.max():.3f}]")

    # 3. Linear: 用 2026-06-26 单天 OOF
    print("\n[Linear] using 2026-06-26 single day ...", flush=True)
    linear_dir = base_dir / 'linear_regression'
    lin_test_files = sorted([f for f in os.listdir(linear_dir) if f.startswith('oof_scores_seed')])
    lin_avg_all_seeds = np.zeros((np.load(linear_dir / lin_test_files[0]).shape[0], 300), dtype=np.float32)
    for f in lin_test_files:
        lin_avg_all_seeds += np.load(linear_dir / f)
    lin_avg_all_seeds /= len(lin_test_files)
    last_day_lin = lin_avg_all_seeds[test_last_idx]
    z_lin_test = np.tile(zscore_per_day(last_day_lin[None, :])[0][None, :], (len(forecast_dates), 1))
    print(f"  Linear (2026-06-26) z-score range=[{z_lin_test[0].min():.3f}, {z_lin_test[0].max():.3f}]")

    # 4. Explicit signals: 用 2026-06-26 单天
    print("\n[Signals] using 2026-06-26 single day ...", flush=True)
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    df['past_ret_5d'] = (df['收盘'] / df.groupby('股票代码')['收盘'].shift(5) - 1).clip(-0.3, 0.3)

    idx = pd.read_csv(INDEX_CSV, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    idx['_d_str'] = idx['date'].dt.strftime('%Y-%m-%d')
    idx['log_ret'] = np.log(idx['close'] / idx['close'].shift(1)).fillna(0)
    hs300_ret_map = dict(zip(idx['_d_str'], idx['log_ret']))

    df['log_ret'] = np.log(df['收盘'] / df['收盘'].shift(1)).fillna(0)
    df['hs300_ret'] = df['日期'].dt.strftime('%Y-%m-%d').map(hs300_ret_map).fillna(0)
    df['_d_str'] = df['日期'].dt.strftime('%Y-%m-%d')

    print("[Signals] computing 60d rolling beta vs HS300 ...", flush=True)
    df['beta_60'] = (df.groupby('股票代码', sort=False)
                      .apply(lambda g: g['log_ret'].rolling(60, min_periods=20).corr(g['hs300_ret']),
                             include_groups=False)
                      .reset_index(level=0, drop=True))

    n_days, n_stocks = len(forecast_dates), len(all_stocks)
    stock_to_idx = {int(s): i for i, s in enumerate(all_stocks)}

    def pivot_single(col):
        mat = np.zeros(n_stocks, dtype=np.float32)
        sub = df[df['_d_str'] == last_day][['股票代码', col]].copy()
        for _, row in sub.iterrows():
            s = int(row['股票代码'])
            if s in stock_to_idx:
                v = row[col]
                if pd.notna(v):
                    mat[stock_to_idx[s]] = float(v)
        return mat

    past_ret_5d_last = pivot_single('past_ret_5d')
    beta_60_last = pivot_single('beta_60')
    test_signals = {
        'reversal_score': np.tile((-past_ret_5d_last)[None, :], (n_days, 1)),
        'momentum_score': np.tile(past_ret_5d_last[None, :], (n_days, 1)),
        'low_beta_score': np.tile((-beta_60_last.clip(-1, 1))[None, :], (n_days, 1)),
        'high_beta_score': np.tile(beta_60_last.clip(-1, 1)[None, :], (n_days, 1)),
        'diversified_score': np.zeros((n_days, n_stocks), dtype=np.float32),
        'macro_industry_beta': np.tile(beta_60_last[None, :], (n_days, 1)),
    }
    print(f"  past_ret_5d range=[{past_ret_5d_last.min():.3f}, {past_ret_5d_last.max():.3f}]")
    print(f"  beta_60 range=[{beta_60_last.min():.3f}, {beta_60_last.max():.3f}]")

    # 5. Regime classifier: 2026-06-26 单天 regime (5 个 forecast dates 共享)
    print("\n[Regime] using 2026-06-26 single day ...", flush=True)
    from regime_classifier import RegimeClassifier
    rc = RegimeClassifier()
    last_1_regime = rc.predict([last_day])[0]
    test_regime = np.repeat(last_1_regime[0], n_days)  # 5 个 forecast dates 共享
    from collections import Counter
    dist = Counter(test_regime.tolist())
    print(f"  {last_day} regime: {REGIME_NAMES_LOC[int(last_1_regime[0])]} ({dist.get(0,0)} bear, "
          f"{dist.get(1,0)} side, {dist.get(2,0)} bull)")

    # 6. Load 3 Stack models
    print("\n[Stack] loading 3 regime-specific models ...", flush=True)
    ensemble_model_dir = MODEL_INTEGRATED_DIR / 'ensemble'
    regime_models = {}
    for r_name in ['bear', 'sideways', 'bull']:
        pkl_p = ensemble_model_dir / f'stack_{r_name}.pkl'
        if pkl_p.exists():
            regime_models[r_name] = joblib.load(pkl_p)
            print(f"  loaded stack_{r_name}.pkl", flush=True)

    # 7. Stack inference
    n_features_per_regime = {0: 6, 1: 6, 2: 6}
    final_pred = stack_inference(regime_models, z_regime_oofs, z_lin_test, test_signals,
                                  test_regime, test_macro, industry_map, all_stocks,
                                  forecast_dates, n_features_per_regime)
    print(f"\n[Stack] forecast inference: shape={final_pred.shape}, "
          f"range=[{final_pred.min():.4f}, {final_pred.max():.4f}]", flush=True)

    # 8. Top-5 industry_diversified picks per day
    print("\n[Top5] selecting top-5 industry_diversified picks ...", flush=True)
    industries = sorted(set(industry_map.values()))
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    stock_industry = np.array([
        industry_to_idx.get(industry_map.get(int(s), ''), 0)
        for s in all_stocks
    ])
    stockid2idx = {int(s): i for i, s in enumerate(all_stocks)}

    all_picks = []  # [(date, stock_id, score), ...]
    for d_idx, d_str in enumerate(forecast_dates):
        scores = final_pred[d_idx]
        sorted_idx = np.argsort(-scores)  # descending
        # Regime-conditional selection (8 组合实验最优):
        #   bear (0) + bull (2): top5_ic (无行业约束, alpha 自由表达)
        #   sideways (1): industry_diversified (行业分散, 避免集中)
        d_regime = int(test_regime[d_idx]) if len(test_regime) > d_idx else 1
        if d_regime in (0, 2):  # bear / bull: top5_ic
            picked = [int(all_stocks[s_idx]) for s_idx in sorted_idx[:5]]
        else:  # sideways: industry_diversified
            picked = []
            picked_industries = set()
            for s_idx in sorted_idx:
                if len(picked) >= 5:
                    break
                sid = stock_industry[s_idx]
                if sid not in picked_industries:
                    picked.append(int(all_stocks[s_idx]))
                    picked_industries.add(sid)
            if len(picked) < 5:
                for s_idx in sorted_idx:
                    if int(all_stocks[s_idx]) not in picked:
                        picked.append(int(all_stocks[s_idx]))
                    if len(picked) >= 5:
                        break
        for i, stock_id in enumerate(picked):
            all_picks.append({
                'date': d_str,
                'rank': i + 1,
                'stock_id': stock_id,
                'score': float(scores[stockid2idx[stock_id]]),
            })
    picks_df = pd.DataFrame(all_picks)
    print(f"  total picks: {len(picks_df)} ({len(forecast_dates)} days × 5 picks)")

    # 9. Generate result.csv
    print("\n[Result] generating result.csv (5 stocks, equal weight 0.2 each, first day picks) ...", flush=True)

    first_day = forecast_dates[0]
    first_day_picks = picks_df[picks_df['date'] == first_day].sort_values('rank')
    print(f"  First day ({first_day}) top-5 picks:")
    print(first_day_picks[['rank', 'stock_id', 'score']].to_string(index=False))

    result = []
    n_picks = len(first_day_picks)
    weight = 1.0 / n_picks if n_picks > 0 else 0
    for _, row in first_day_picks.iterrows():
        result.append({'stock_id': int(row['stock_id']), 'weight': weight})

    result_df = pd.DataFrame(result)
    out_path = base_dir / 'result.csv'
    result_df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\n[Save] {out_path}:")
    print(result_df.to_string(index=False))
    print(f"\n[Format check]:")
    with open(out_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"  Header: {lines[0].strip()}")
    for line in lines[1:]:
        print(f"  Row: {line.strip()}")
    print(f"  Sum of weights: {result_df['weight'].sum():.4f} (<= 1.0)")


REGIME_NAMES_LOC = ['bear', 'sideways', 'bull']


if __name__ == '__main__':
    main()
