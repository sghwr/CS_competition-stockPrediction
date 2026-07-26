"""v1.0 Ensemble: Regime-Specific 8 Models with 显式策略.

架构:
  - MacroTransformer (1 model): industry_bias + industry_beta (industry 级别)
  - Tree (3 models): bull/sideways/bear, 各自训在对应 regime data
  - Linear (1 model): 短期反转 (regime 无关, 10 dim)
  - Stack (3 models): bull/sideways/bear, 各自用 **regime-specific 显式 features**

显式策略 features (per regime stack):
  - bear_stack: tree + linear + macro_bias + industry_id + reversal + low_beta
                (选输家反弹 + 低 beta, defensive)
  - sideways_stack: tree + linear + macro_bias + industry_id + macro_industry_beta
                (行业 beta 中性, balanced)
  - bull_stack: tree + linear + macro_bias + industry_id + reversal + low_beta
                (跟 bear 镜像, 8 组合实验证明 bull 用 reversal+low_beta 显著 work)

推理:
  - regime_pred (300-equal past_5d, deterministic)
  - 选对应 tree + stack model
  - 输出最终预测
"""
import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TRAIN_CSV, MODEL_INTEGRATED_DIR
from config import config
from industry_mapping import load_industry_map

REGIME_NAMES = ['bear', 'sideways', 'bull']

# 公共 base features (3 个 stack models 都用)
COMMON_BASE_FEATURES = [
    'tree_pred',            # regime-specific Tree OOF
    'linear_pred',          # Linear 10 dim 反转
    'macro_industry_bias',  # Macro industry alpha (per-stock)
    'industry_id',          # categorical [0-10]
]

# Regime-specific 显式策略 features
# 关键: 每个 regime 显式学自己的策略, 不是仅仅切分训练数据
REGIME_STRATEGY_FEATURES = {
    0: ['reversal_score', 'low_beta_score'],          # bear: 选输家反弹 + 低 beta
    1: ['macro_industry_beta'],                       # sideways: 行业 beta 中性
    2: ['reversal_score', 'low_beta_score'],          # bull: 反转 + 低 beta (跟 bear 镜像)
}

# 全 features per regime (build_stack_features 输出)
STACK_FEATURES_PER_REGIME = {
    r: COMMON_BASE_FEATURES + REGIME_STRATEGY_FEATURES[r]
    for r in [0, 1, 2]
}

# 列数 (每个 regime 不同, bear/side/bull 都是 6 dim)
STACK_N_FEATURES = {r: len(STACK_FEATURES_PER_REGIME[r]) for r in [0, 1, 2]}


def load_oof_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def load_macro_pred(base_dir, fallback_macro_dir=None):
    """Load macro_pred_avg.npz or fallback to single seed.

    Returns: industry_bias + industry_beta (no regime).
    """
    macro_dir = os.path.join(base_dir, 'macro_transformer')
    if not os.path.exists(macro_dir) and fallback_macro_dir:
        macro_dir = fallback_macro_dir
    avg_path = os.path.join(macro_dir, 'macro_pred_avg.npz')
    if os.path.exists(avg_path):
        npz = np.load(avg_path, allow_pickle=True)
        print(f"  [Load] using macro_pred_avg.npz (multi-seed)", flush=True)
    else:
        files = sorted([f for f in os.listdir(macro_dir) if f.startswith('macro_pred_seed')])
        if not files:
            raise FileNotFoundError(f"No macro_pred_seed*.npz in {macro_dir}")
        npz = np.load(os.path.join(macro_dir, files[0]), allow_pickle=True)
        print(f"  [Load] using {files[0]} (single seed) from {macro_dir}", flush=True)
    return {
        'dates': list(npz['dates']),
        'industry_bias': npz['industry_bias'],
        'industry_beta': npz['industry_beta'] if 'industry_beta' in npz.files else None,
        'industries': list(npz['industries']),
    }


def load_val_meta(base_dir):
    """Load val_meta.json (6-month validation period)."""
    candidates = [
        os.path.join(base_dir, 'tree', 'val_meta.json'),
        os.path.join(base_dir, 'linear_regression', 'val_meta.json'),
        os.path.join(base_dir, 'macro_transformer', 'val_meta.json'),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                meta = json.load(f)
            print(f"  [Load] val_meta from {p}: {len(meta['dates'])} dates "
                  f"({meta['dates'][0]} ~ {meta['dates'][-1]})", flush=True)
            return meta
    raise FileNotFoundError(f"No val_meta.json found in {candidates}")


def load_test_meta(base_dir, fallback_macro_dir=None):
    p = os.path.join(base_dir, 'macro_transformer', 'test_meta.json')
    if not os.path.exists(p) and fallback_macro_dir:
        p = os.path.join(fallback_macro_dir, 'test_meta.json')
    with open(p, encoding='utf-8') as f:
        meta = json.load(f)
    print(f"  [Load] test_meta from {p}: {len(meta['dates'])} dates "
          f"({meta['dates'][0]} ~ {meta['dates'][-1]})", flush=True)
    return meta


def load_train_meta(base_dir):
    """Load train_meta.json (10-year training period)."""
    candidates = [
        os.path.join(base_dir, 'tree', 'train_meta.json'),
        os.path.join(base_dir, 'linear_regression', 'train_meta.json'),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                meta = json.load(f)
            print(f"  [Load] train_meta from {p}: {len(meta['dates'])} dates "
                  f"({meta['dates'][0]} ~ {meta['dates'][-1]})", flush=True)
            return meta
    raise FileNotFoundError(f"No train_meta.json found in {candidates}")


def get_y_lookup(dates_set):
    """Compute 5d forward returns for given dates."""
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['y_5d'] = ((df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)).clip(-0.3, 0.3)
    val = df[df['日期'].dt.strftime('%Y-%m-%d').isin(dates_set)].copy()
    val['日期_str'] = val['日期'].dt.strftime('%Y-%m-%d')
    y_lookup = {}
    for _, row in val[['日期_str', '股票代码', 'y_5d']].iterrows():
        y_lookup[(row['日期_str'], int(row['股票代码']))] = row['y_5d']
    return y_lookup


def build_explicit_signals(dates, all_stocks):
    """Build 4 regime-specific explicit signals per stock.

    NO LEAK: signals 用 **past** data (不用 future).
      - reversal_score: -past_ret_5d (输家反弹, A 股 5d 反转市场)
      - momentum_score: +past_ret_5d (赢家延续, 牛市用)
      - low_beta_score:  -beta_60 (低 beta, defensive, 60d rolling corr vs HS300)
      - high_beta_score: +beta_60 (高 beta, aggressive)
      - diversified_score: 0 (constant, model 自适应)
    """
    print("[Signals] building 4 explicit regime-specific signals (reversal/momentum/beta, NO LEAK) ...",
          flush=True)
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # ⚠️ NO LEAK: 用 **past** ret_5d (close[t]/close[t-5] - 1, 不含 future)
    df['past_ret_5d'] = (df['收盘'] / df.groupby('股票代码')['收盘'].shift(5) - 1).clip(-0.3, 0.3)

    # 60d rolling corr (beta) vs HS300 daily return
    idx = pd.read_csv('data/index_data.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    idx['_d_str'] = idx['date'].dt.strftime('%Y-%m-%d')
    idx['log_ret'] = np.log(idx['close'] / idx['close'].shift(1)).fillna(0)
    hs300_ret_map = dict(zip(idx['_d_str'], idx['log_ret']))

    # Stock 60d rolling beta vs HS300 (用 past returns, no future)
    df['log_ret'] = np.log(df['收盘'] / df['收盘'].shift(1)).fillna(0)
    df['hs300_ret'] = df['日期'].dt.strftime('%Y-%m-%d').map(hs300_ret_map).fillna(0)
    df['_d_str'] = df['日期'].dt.strftime('%Y-%m-%d')

    # Per stock rolling 60d corr
    print("[Signals] computing 60d rolling beta vs HS300 (per stock) ...", flush=True)
    df['beta_60'] = (df.groupby('股票代码', sort=False)
                      .apply(lambda g: g['log_ret'].rolling(60, min_periods=20).corr(g['hs300_ret']),
                             include_groups=False)
                      .reset_index(level=0, drop=True))

    # Pivot to (n_days, n_stocks) matrix
    date_to_idx = {d: i for i, d in enumerate(dates)}
    stock_to_idx = {int(s): i for i, s in enumerate(all_stocks)}
    n_days, n_stocks = len(dates), len(all_stocks)

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

    # 4 个显式 signals (NO LEAK)
    reversal_score = -past_ret_5d  # 输家反弹 (past)
    momentum_score = past_ret_5d    # 赢家延续 (past)
    low_beta_score = -beta_60.clip(-1, 1)
    high_beta_score = beta_60.clip(-1, 1)
    diversified_score = np.zeros((n_days, n_stocks), dtype=np.float32)

    print(f"  reversal_score: range=[{reversal_score.min():.4f}, {reversal_score.max():.4f}], "
          f"mean={reversal_score.mean():.4f}", flush=True)
    print(f"  momentum_score: range=[{momentum_score.min():.4f}, {momentum_score.max():.4f}], "
          f"mean={momentum_score.mean():.4f}", flush=True)
    print(f"  low_beta_score: range=[{low_beta_score.min():.4f}, {low_beta_score.max():.4f}], "
          f"mean={low_beta_score.mean():.4f}", flush=True)
    print(f"  high_beta_score: range=[{high_beta_score.min():.4f}, {high_beta_score.max():.4f}], "
          f"mean={high_beta_score.mean():.4f}", flush=True)

    return {
        'reversal_score': reversal_score,
        'momentum_score': momentum_score,
        'low_beta_score': low_beta_score,
        'high_beta_score': high_beta_score,
        'diversified_score': diversified_score,
        'macro_industry_beta': beta_60,
    }


def build_stack_features(tree_oof, linear_oof, dates, all_stocks, macro_pred, industry_map,
                          signals=None, regime=None):
    """Build stack features (per regime, 显式策略).

      bear_stack (5+1): tree + linear + macro_bias + industry_id + reversal + low_beta
      sideways_stack (4+1): tree + linear + macro_bias + industry_id + macro_industry_beta
      bull_stack (5+1): tree + linear + macro_bias + industry_id + reversal + low_beta
    """
    n_days = len(dates)
    n_stocks = len(all_stocks)

    industries = sorted(set(industry_map.values()))
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    stock_industry = np.array([
        industry_to_idx.get(industry_map.get(int(s), ''), 0)
        for s in all_stocks
    ], dtype=np.int32)

    macro_date_to_idx = {d: i for i, d in enumerate(macro_pred['dates'])}
    macro_industries = macro_pred['industries']
    macro_ind_to_idx = {ind: i for i, ind in enumerate(macro_industries)}
    full_to_macro = np.array([macro_ind_to_idx.get(ind, -1) for ind in industries], dtype=np.int32)
    valid_macro = full_to_macro >= 0

    if regime is None:
        features = COMMON_BASE_FEATURES
    else:
        features = STACK_FEATURES_PER_REGIME[regime]
    n_features = len(features)
    X = np.zeros((n_days * n_stocks, n_features), dtype=np.float32)

    # Pre-compute macro arrays
    macro_bias_mat = np.zeros((n_days, len(industries)), dtype=np.float32)
    macro_beta_mat = np.zeros((n_days, len(industries)), dtype=np.float32)
    for d_idx, d_str in enumerate(dates):
        m_idx = macro_date_to_idx.get(d_str)
        if m_idx is not None and 'industry_bias' in macro_pred:
            macro_bias = macro_pred['industry_bias'][m_idx]
            for i in range(len(industries)):
                if valid_macro[i]:
                    macro_bias_mat[d_idx, i] = macro_bias[full_to_macro[i]]
        if m_idx is not None and 'industry_beta' in macro_pred:
            macro_beta = macro_pred['industry_beta'][m_idx]
            for i in range(len(industries)):
                if valid_macro[i]:
                    macro_beta_mat[d_idx, i] = macro_beta[full_to_macro[i]]

    for d_idx in range(n_days):
        for s_idx in range(n_stocks):
            sample_idx = d_idx * n_stocks + s_idx
            sid = stock_industry[s_idx]
            t_pred = tree_oof[d_idx, s_idx] if not np.isnan(tree_oof[d_idx, s_idx]) else 0
            l_pred = linear_oof[d_idx, s_idx] if not np.isnan(linear_oof[d_idx, s_idx]) else 0
            col = 0
            X[sample_idx, col] = t_pred; col += 1
            X[sample_idx, col] = l_pred; col += 1
            X[sample_idx, col] = macro_bias_mat[d_idx, sid]; col += 1
            X[sample_idx, col] = sid; col += 1
            # regime-specific signals
            if regime is not None and signals is not None:
                for sig_name in REGIME_STRATEGY_FEATURES[regime]:
                    sig_mat = signals[sig_name]
                    val = sig_mat[d_idx, s_idx] if d_idx < sig_mat.shape[0] else 0
                    X[sample_idx, col] = val; col += 1

    return X


def get_regime_for_dates(dates):
    """Get 3-class regime (0=bear, 1=sideways, 2=bull) for each date."""
    from regime_classifier import RegimeClassifier
    rc = RegimeClassifier()
    regime_pred, _ = rc.predict(dates)
    return regime_pred


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


def build_stack_targets(dates, all_stocks, y_lookup):
    n_days = len(dates)
    n_stocks = len(all_stocks)
    y = np.full((n_days, n_stocks), np.nan, dtype=np.float32)
    for d_idx, d_str in enumerate(dates):
        for s_idx, s in enumerate(all_stocks):
            y[d_idx, s_idx] = y_lookup.get((d_str, int(s)), np.nan)
    return y


def compute_daily_ic(pred, y_mat):
    """Compute per-day Spearman IC, return mean."""
    ics = []
    n_days = pred.shape[0]
    for d in range(n_days):
        trues = y_mat[d]
        preds = pred[d]
        mask = ~np.isnan(trues) & ~np.isnan(preds)
        if mask.sum() > 30 and np.std(preds[mask]) > 1e-9:
            ic, _ = spearmanr(preds[mask], trues[mask])
            if not np.isnan(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0, ics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True,
                        help='推理产物根目录 (output/integrated_v1)')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='模型权重目录 (model/integrated_v1/ensemble), 默认 = MODEL_INTEGRATED_DIR/ensemble')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认 = base_dir/ensemble)')
    parser.add_argument('--fallback_macro_dir', type=str, default=None)
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = str(MODEL_INTEGRATED_DIR / 'ensemble')
    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_dir, 'ensemble')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load meta
    print("\n[Load] train_meta (10 年, 用于 stack 训练) ...", flush=True)
    train_meta = load_train_meta(args.base_dir)
    print("[Load] val_meta (6 月, 用于 stack OOS 评估) ...", flush=True)
    val_meta = load_val_meta(args.base_dir)
    print("[Load] test_meta (6 月, 终极 OOS) ...", flush=True)
    test_meta = load_test_meta(args.base_dir, args.fallback_macro_dir)

    # 2. Load all
    print("\n[Load] MacroTransformer (industry alpha + beta) ...", flush=True)
    macro_pred = load_macro_pred(args.base_dir, args.fallback_macro_dir)
    print(f"  industry_bias range=[{macro_pred['industry_bias'].min():.4f}, "
          f"{macro_pred['industry_bias'].max():.4f}]", flush=True)

    # Load 3 regime-specific Tree OOF (train/val/test)
    print("\n[Load] Tree (3 regime-specific OOF) ...", flush=True)
    tree_train_oofs = {}
    tree_val_oofs = {}
    tree_test_oofs = {}
    for r, r_name in enumerate(REGIME_NAMES):
        train_p = os.path.join(args.base_dir, 'tree', f'train_oof_{r_name}.npy')
        val_p = os.path.join(args.base_dir, 'tree', f'val_oof_{r_name}.npy')
        test_p = os.path.join(args.base_dir, 'tree', f'oof_scores_{r_name}.npy')
        if os.path.exists(val_p) and os.path.exists(test_p) and os.path.exists(train_p):
            tree_train_oofs[r] = load_oof_npy(train_p)
            tree_val_oofs[r] = load_oof_npy(val_p)
            tree_test_oofs[r] = load_oof_npy(test_p)
            print(f"  tree_{r_name}: train {tree_train_oofs[r].shape}, "
                  f"val {tree_val_oofs[r].shape}, test {tree_test_oofs[r].shape}", flush=True)
    if not tree_val_oofs:
        raise FileNotFoundError("No regime-specific Tree OOF found, run tree_branch.py first")

    print("\n[Load] LinearRegression train/val/test OOF ...", flush=True)
    lin_train_files = sorted([f for f in os.listdir(os.path.join(args.base_dir, 'linear_regression'))
                              if f.startswith('train_oof_seed')])
    lin_val_files = sorted([f for f in os.listdir(os.path.join(args.base_dir, 'linear_regression'))
                            if f.startswith('val_oof_seed')])
    lin_test_files = sorted([f for f in os.listdir(os.path.join(args.base_dir, 'linear_regression'))
                             if f.startswith('oof_scores_seed')])
    lin_train = np.mean([load_oof_npy(os.path.join(args.base_dir, 'linear_regression', f))
                         for f in lin_train_files], axis=0) if lin_train_files else None
    lin_val = np.mean([load_oof_npy(os.path.join(args.base_dir, 'linear_regression', f))
                       for f in lin_val_files], axis=0) if lin_val_files else None
    lin_test = np.mean([load_oof_npy(os.path.join(args.base_dir, 'linear_regression', f))
                        for f in lin_test_files], axis=0) if lin_test_files else None
    print(f"  linear train files: {lin_train_files}, val files: {lin_val_files}, test files: {lin_test_files}", flush=True)
    if lin_train is None or lin_val is None or lin_test is None:
        raise FileNotFoundError("Linear OOF (train/val/test) not found")
    print(f"  linear train: {lin_train.shape}, val: {lin_val.shape}, test: {lin_test.shape}", flush=True)

    all_stocks = sorted(pd.read_csv(TRAIN_CSV, encoding='utf-8-sig',
                                     usecols=['股票代码'])['股票代码'].unique())
    industry_map = load_industry_map()

    # 3. Regime labels for all 3 periods
    print("\n[Regime] computing 3-class regime (300-equal past_5d, deterministic) ...", flush=True)
    train_regime = get_regime_for_dates(train_meta['dates'])
    val_regime = get_regime_for_dates(val_meta['dates'])
    test_regime = get_regime_for_dates(test_meta['dates'])
    from collections import Counter
    train_dist = Counter(train_regime.tolist())
    val_dist = Counter(val_regime.tolist())
    test_dist = Counter(test_regime.tolist())
    print(f"  train: bear={train_dist.get(0,0)}, side={train_dist.get(1,0)}, bull={train_dist.get(2,0)}",
          flush=True)
    print(f"  val: bear={val_dist.get(0,0)}, side={val_dist.get(1,0)}, bull={val_dist.get(2,0)}",
          flush=True)
    print(f"  test: bear={test_dist.get(0,0)}, side={test_dist.get(1,0)}, bull={test_dist.get(2,0)}",
          flush=True)

    # 4. Build features
    print("\n[Build] routed tree OOF (per regime) + features (5/6 dim per regime) ...", flush=True)
    z_tree_train_oofs = {r: zscore_per_day(oof) for r, oof in tree_train_oofs.items()}
    z_tree_val_oofs = {r: zscore_per_day(oof) for r, oof in tree_val_oofs.items()}
    z_tree_test_oofs = {r: zscore_per_day(oof) for r, oof in tree_test_oofs.items()}
    z_lin_train = zscore_per_day(lin_train)
    z_lin_val = zscore_per_day(lin_val)
    z_lin_test = zscore_per_day(lin_test)
    train_regime_repeat = np.repeat(train_regime, len(all_stocks))
    val_regime_repeat = np.repeat(val_regime, len(all_stocks))
    test_regime_repeat = np.repeat(test_regime, len(all_stocks))

    # Explicit regime-specific signals
    train_signals = build_explicit_signals(train_meta['dates'], all_stocks)
    val_signals = build_explicit_signals(val_meta['dates'], all_stocks)
    test_signals = build_explicit_signals(test_meta['dates'], all_stocks)

    # Pre-compute macro & industry mappings
    industries = sorted(set(industry_map.values()))
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    stock_industry = np.array([
        industry_to_idx.get(industry_map.get(int(s), ''), 0)
        for s in all_stocks
    ], dtype=np.int32)
    macro_date_to_idx = {d: i for i, d in enumerate(macro_pred['dates'])}
    macro_industries = macro_pred['industries']
    macro_ind_to_idx = {ind: i for i, ind in enumerate(macro_industries)}
    full_to_macro = np.array([macro_ind_to_idx.get(ind, -1) for ind in industries], dtype=np.int32)
    valid_macro = full_to_macro >= 0

    def build_X_for_regime(regime, dates, regime_repeat, tree_oofs, lin_oof, signals):
        n_features = STACK_N_FEATURES[regime]
        n_days = len(dates)
        n_stocks = len(all_stocks)
        X = np.zeros((n_days * n_stocks, n_features), dtype=np.float32)
        macro_bias_mat = np.zeros((n_days, len(industries)), dtype=np.float32)
        for d_idx, d_str in enumerate(dates):
            m_idx = macro_date_to_idx.get(d_str)
            if m_idx is not None and 'industry_bias' in macro_pred:
                macro_bias = macro_pred['industry_bias'][m_idx]
                for i in range(len(industries)):
                    if valid_macro[i]:
                        macro_bias_mat[d_idx, i] = macro_bias[full_to_macro[i]]
        for d_idx, d_str in enumerate(dates):
            r = int(regime_repeat[d_idx])
            tree_oof_d = tree_oofs.get(r, tree_oofs.get(1))
            for s_idx in range(n_stocks):
                sample_idx = d_idx * n_stocks + s_idx
                sid = stock_industry[s_idx]
                t_pred = tree_oof_d[d_idx, s_idx] if not np.isnan(tree_oof_d[d_idx, s_idx]) else 0
                l_pred = lin_oof[d_idx, s_idx] if not np.isnan(lin_oof[d_idx, s_idx]) else 0
                col = 0
                X[sample_idx, col] = t_pred; col += 1
                X[sample_idx, col] = l_pred; col += 1
                X[sample_idx, col] = macro_bias_mat[d_idx, sid]; col += 1
                X[sample_idx, col] = sid; col += 1
                for sig_name in REGIME_STRATEGY_FEATURES[regime]:
                    sig_mat = signals[sig_name]
                    val_ = sig_mat[d_idx, s_idx] if d_idx < sig_mat.shape[0] else 0
                    X[sample_idx, col] = val_; col += 1
        return X

    X_train_per_regime = {}
    X_val_per_regime = {}
    X_test_per_regime = {}
    for r in [0, 1, 2]:
        X_train_per_regime[r] = build_X_for_regime(r, train_meta['dates'], train_regime,
                                                    z_tree_train_oofs, z_lin_train, train_signals)
        X_val_per_regime[r] = build_X_for_regime(r, val_meta['dates'], val_regime,
                                                  z_tree_val_oofs, z_lin_val, val_signals)
        X_test_per_regime[r] = build_X_for_regime(r, test_meta['dates'], test_regime,
                                                   z_tree_test_oofs, z_lin_test, test_signals)
        print(f"  X_train_{REGIME_NAMES[r]}: {X_train_per_regime[r].shape}, "
              f"X_val: {X_val_per_regime[r].shape}, X_test: {X_test_per_regime[r].shape}", flush=True)

    # 5. Targets
    print("\n[Targets] building y_5d ...", flush=True)
    y_train_mat = build_stack_targets(train_meta['dates'], all_stocks,
                                       get_y_lookup(set(train_meta['dates'])))
    y_val_mat = build_stack_targets(val_meta['dates'], all_stocks,
                                     get_y_lookup(set(val_meta['dates'])))
    y_test_mat = build_stack_targets(test_meta['dates'], all_stocks,
                                      get_y_lookup(set(test_meta['dates'])))
    print(f"  y_train valid: {(~np.isnan(y_train_mat)).sum():,}/{y_train_mat.size:,}", flush=True)
    print(f"  y_val valid: {(~np.isnan(y_val_mat)).sum():,}/{y_val_mat.size:,}", flush=True)
    print(f"  y_test valid: {(~np.isnan(y_test_mat)).sum():,}/{y_test_mat.size:,}", flush=True)

    # 6. Train 3 regime-specific stack models on TRAIN only
    print("\n[Stack] training 3 regime-specific LightGBM (5/6 dim per regime, 训在 TRAIN) ...", flush=True)
    import lightgbm as lgb

    y_train_flat = y_train_mat.flatten()
    y_val_flat = y_val_mat.flatten()
    y_test_flat = y_test_mat.flatten()
    train_mask_valid = ~np.isnan(y_train_flat)
    val_mask_valid = ~np.isnan(y_val_flat)
    test_mask_valid = ~np.isnan(y_test_flat)

    stack_models = {}
    train_preds = np.full(X_train_per_regime[1].shape[0], np.nan, dtype=np.float32)
    val_preds = np.full(X_val_per_regime[1].shape[0], np.nan, dtype=np.float32)
    test_preds = np.full(X_test_per_regime[1].shape[0], np.nan, dtype=np.float32)

    for r in [0, 1, 2]:
        r_name = REGIME_NAMES[r]
        train_mask_r = train_regime_repeat == r
        train_mask_fit = train_mask_valid & train_mask_r
        n_train_r = train_mask_fit.sum()

        if n_train_r < 100:
            print(f"  [WARN] {r_name}: only {n_train_r} train samples, skip (use sideways fallback)")
            continue

        X_fit_r = X_train_per_regime[r][train_mask_fit]
        y_fit_r = y_train_flat[train_mask_fit]
        print(f"  [{r_name}] train samples: {len(y_fit_r):,} (TRAIN set), "
              f"features: {STACK_FEATURES_PER_REGIME[r]}", flush=True)

        model_r = lgb.LGBMRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            num_leaves=7, min_data_in_leaf=100,
            reg_alpha=0.1, reg_lambda=0.1, verbose=-1,
            random_state=42,
        )
        model_r.fit(
            X_fit_r, y_fit_r,
            categorical_feature=[STACK_FEATURES_PER_REGIME[r].index('industry_id')],
        )
        stack_models[r] = model_r

        # predict on train (in-sample)
        train_mask_r_all = train_mask_r
        if train_mask_r_all.sum() > 0:
            train_preds[train_mask_r_all] = model_r.predict(X_train_per_regime[r][train_mask_r_all])
        # predict on val (OOS)
        val_mask_r = val_regime_repeat == r
        if val_mask_r.sum() > 0:
            val_preds[val_mask_r] = model_r.predict(X_val_per_regime[r][val_mask_r])
        # predict on test (OOS)
        test_mask_r = test_regime_repeat == r
        if test_mask_r.sum() > 0:
            test_preds[test_mask_r] = model_r.predict(X_test_per_regime[r][test_mask_r])

    fallback_model = stack_models.get(1) or stack_models.get(2) or stack_models.get(0)
    for r in [0, 1, 2]:
        if r not in stack_models:
            for mask_set, preds_arr, X_dict in [
                (train_regime_repeat == r, train_preds, X_train_per_regime),
                (val_regime_repeat == r, val_preds, X_val_per_regime),
                (test_regime_repeat == r, test_preds, X_test_per_regime),
            ]:
                if mask_set.sum() > 0:
                    preds_arr[mask_set] = fallback_model.predict(X_dict[r][mask_set])

    final_train = train_preds.reshape(len(train_meta['dates']), len(all_stocks))
    final_val = val_preds.reshape(len(val_meta['dates']), len(all_stocks))
    final_test = test_preds.reshape(len(test_meta['dates']), len(all_stocks))

    train_ic, _ = compute_daily_ic(final_train, y_train_mat)
    val_ic, _ = compute_daily_ic(final_val, y_val_mat)
    test_ic, _ = compute_daily_ic(final_test, y_test_mat)
    print(f"\n[Regime-Stack] train IC: {train_ic:+.4f} (训集, 仅供参考)", flush=True)
    print(f"[Regime-Stack] val IC:   {val_ic:+.4f} (OOS, 真实验证指标)", flush=True)
    print(f"[Regime-Stack] test IC:  {test_ic:+.4f} (OOS, 真实终测指标)", flush=True)

    print("\n[Regime-Stack] per-regime IC:", flush=True)
    for period_name, final_arr, regime_arr in [
        ('val', final_val, val_regime),
        ('test', final_test, test_regime),
    ]:
        print(f"  [{period_name}]", flush=True)
        for r in [0, 1, 2]:
            r_name = REGIME_NAMES[r]
            mask = regime_arr == r
            if mask.sum() > 5:
                ic, _ = compute_daily_ic(final_arr[mask], y_val_mat[mask]
                                          if period_name == 'val' else y_test_mat[mask])
                print(f"    {r_name:9s} ({mask.sum():>3} days): IC={ic:+.4f}", flush=True)

    # Per-component test IC
    print("\n[Per-component] test IC:", flush=True)
    z_routed_tree_test = np.zeros_like(z_lin_test)
    for d_idx in range(len(test_meta['dates'])):
        r = int(test_regime[d_idx])
        oof = z_tree_test_oofs.get(r, z_tree_test_oofs.get(1))
        z_routed_tree_test[d_idx] = oof[d_idx]
    tree_ic, _ = compute_daily_ic(z_routed_tree_test, y_test_mat)
    lin_ic, _ = compute_daily_ic(z_lin_test, y_test_mat)
    ib_test = np.zeros((len(test_meta['dates']), len(all_stocks)), dtype=np.float32)
    for d_idx, d_str in enumerate(test_meta['dates']):
        m_idx = macro_date_to_idx.get(d_str)
        if m_idx is None: continue
        macro_bias = macro_pred['industry_bias'][m_idx]
        bias_d = np.zeros(len(industries), dtype=np.float32)
        for i in range(len(industries)):
            if valid_macro[i]:
                bias_d[i] = macro_bias[full_to_macro[i]]
        ib_test[d_idx] = bias_d[stock_industry]
    macro_ic, _ = compute_daily_ic(ib_test, y_test_mat)
    print(f"  routed tree={tree_ic:+.4f}, linear={lin_ic:+.4f}, macro={macro_ic:+.4f}", flush=True)

    print("\n[Stack Model] feature importance (per regime, 显式策略):", flush=True)
    for r in [0, 1, 2]:
        if r in stack_models:
            fi = stack_models[r].feature_importances_
            print(f"  {REGIME_NAMES[r]:9s} (strategy={REGIME_STRATEGY_FEATURES[r]}):", flush=True)
            for name, imp in sorted(zip(STACK_FEATURES_PER_REGIME[r], fi), key=lambda x: -x[1]):
                marker = " (strategy)" if name in REGIME_STRATEGY_FEATURES[r] else ""
                print(f"    {name:30s} {imp:>4d}{marker}", flush=True)

    # Save
    np.save(os.path.join(args.output_dir, 'final_oof_scores.npy'), final_test)
    np.save(os.path.join(args.output_dir, 'val_pred.npy'), final_val)
    for r in [0, 1, 2]:
        if r in stack_models:
            joblib.dump(stack_models[r], os.path.join(args.model_dir, f'stack_{REGIME_NAMES[r]}.pkl'))

    out = {
        'method': 'v1.0_8models_regime_specific_explicit_strategy',
        'architecture': 'Macro (1) + Tree (3 regime-specific) + Linear (1) + Stack (3 regime-specific, 显式策略) = 8 models',
        'stack_features_per_regime': {REGIME_NAMES[r]: STACK_FEATURES_PER_REGIME[r] for r in [0, 1, 2]},
        'regime_strategies': {REGIME_NAMES[r]: REGIME_STRATEGY_FEATURES[r] for r in [0, 1, 2]},
        'regime_distribution': {
            'train': dict(train_dist), 'val': dict(val_dist), 'test': dict(test_dist),
        },
        'train_ic': train_ic,
        'val_ic': val_ic,
        'test_ic': test_ic,
        'per_component_test_ic': {'routed_tree': tree_ic, 'linear': lin_ic, 'macro': macro_ic},
        'splits': config['train_start'] + ' ~ ' + config['train_end'] + ' (train 10 年), ' +
                  config['val_start'] + ' ~ ' + config['val_end'] + ' (val 6 月), ' +
                  config['test_start'] + ' ~ ' + config['test_end'] + ' (test 6 月)',
    }
    with open(os.path.join(args.output_dir, 'weights.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[Save] final_oof_scores.npy (test, strict OOS) + val_pred.npy + "
          f"stack_*.pkl (3 models, 显式策略) + weights.json", flush=True)
    print(f"[Done] train IC = {train_ic:+.4f}, val IC = {val_ic:+.4f}, test IC = {test_ic:+.4f}", flush=True)


if __name__ == '__main__':
    main()
