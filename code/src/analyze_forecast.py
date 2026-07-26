"""Forecast 透明度分析:
1. 市场形态 (regime_classifier 300-equal past_5d)
2. 各 component 原始 OOF (macro, tree×3, linear) for 2026-06-26
3. Stack models 权重 (feature importance) 跟 design
4. 解释 model 选择原因
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import INTEGRATED_DIR, MODEL_INTEGRATED_DIR, TRAIN_CSV, INDEX_CSV
from regime_classifier import RegimeClassifier


def zscore_per_day(arr):
    """Per-day z-score: standardize each row (day) independently."""
    mean = np.nanmean(arr, axis=1, keepdims=True)
    std = np.nanstd(arr, axis=1, keepdims=True)
    return (arr - mean) / (std + 1e-8)


# === 1. 市场形态判断 ===
print("=" * 80)
print("[1] MARKET REGIME (300-equal-weighted past_5d, deterministic)")
print("=" * 80)

rc = RegimeClassifier()
df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
all_dates = sorted(df['日期'].dt.strftime('%Y-%m-%d').unique())
print(f"  Stock data 截止: {all_dates[-1]}, total {len(all_dates)} days")

# 最近 14 天的 regime
last_dates = all_dates[-14:]
regimes, _ = rc.predict(last_dates)
daily = rc.daily_market
last_14 = daily[daily['_d_str'].astype(str).isin(last_dates)].sort_values('日期')
print(f"\n  最近 14 天的市场形态:")
print(f"  {'Date':12s}  {'past_5d':>10s}  Regime")
for _, row in last_14.iterrows():
    d = row['_d_str']
    p5 = row['past_5d']
    if pd.isna(p5):
        continue
    r_name = ['bear', 'sideways', 'bull'][int(rc.predict([d])[0][0])]
    print(f"  {d:12s}  {p5:>+10.4f}  {r_name}")

last_5d = float(last_14['past_5d'].iloc[-1])
last_regime = int(rc.predict([last_14['_d_str'].iloc[-1]])[0][0])
last_day = last_14['_d_str'].iloc[-1]
print(f"\n  >>> Forecast 6.29-7.3 regime = {['bear','sideways','bull'][last_regime]} "
      f"(based on {last_day} 300-equal past_5d = {last_5d:+.4f})")

# Test 段分布对比
test_dates_list = [d for d in all_dates if '2026-01-01' <= d <= '2026-06-26']
test_regimes, _ = rc.predict(test_dates_list)
dist = Counter(int(r) for r in test_regimes)
print(f"\n  test 段 (6 月, 109 days) regime 分布:")
print(f"    bear (0):     {dist[0]:3d} days ({dist[0]/len(test_regimes)*100:5.1f}%)")
print(f"    sideways (1): {dist[1]:3d} days ({dist[1]/len(test_regimes)*100:5.1f}%)")
print(f"    bull (2):     {dist[2]:3d} days ({dist[2]/len(test_regimes)*100:5.1f}%)")


# === 2. 各 component 原始 OOF (2026-06-26) ===
print("\n" + "=" * 80)
print("[2] COMPONENT RAW OOF (2026-06-26, test 段最后 1 天)")
print("=" * 80)

# 2.1 MacroTransformer: industry bias + beta (11 industries × 2)
print("\n[2.1] MacroTransformer (3 features → 11 industry alpha + 11 industry beta)")
macro_npz = np.load(INTEGRATED_DIR / 'macro_transformer' / 'macro_pred_avg.npz', allow_pickle=True)
macro_dates = list(macro_npz['dates'])
last_idx = macro_dates.index('2026-06-26')
industries = list(macro_npz['industries'])
bias = macro_npz['industry_bias'][last_idx]
beta = macro_npz['industry_beta'][last_idx]
print(f"  {'Industry':15s}  {'bias (alpha)':>12s}  {'beta':>8s}")
for i, ind in enumerate(industries):
    print(f"  {ind:15s}  {bias[i]:>+12.4f}  {beta[i]:+8.4f}")
print(f"  Range: bias=[{bias.min():+.4f}, {bias.max():+.4f}], beta=[{beta.min():+.4f}, {beta.max():+.4f}]")

# 2.2 Tree (3 regime): 187 dim stock-level factors
print("\n[2.2] Tree (3 regime-specific, 187 dim stock-level factors)")
all_stocks = sorted(pd.read_csv(TRAIN_CSV, encoding='utf-8-sig',
                                 usecols=['股票代码'])['股票代码'].unique())
test_meta = json.load(open(INTEGRATED_DIR / 'macro_transformer' / 'test_meta.json', encoding='utf-8'))
test_last_idx = test_meta['dates'].index('2026-06-26')
for r_name in ['bear', 'sideways', 'bull']:
    oof = np.load(INTEGRATED_DIR / 'tree' / f'oof_scores_{r_name}.npy')
    day_oof = oof[test_last_idx]  # (300,)
    valid = ~np.isnan(day_oof)
    print(f"  Tree_{r_name:8s}: range=[{day_oof[valid].min():+.4f}, {day_oof[valid].max():+.4f}], "
          f"mean={day_oof[valid].mean():+.4f}, std={day_oof[valid].std():.4f}")

# 2.3 Linear (1): 10 dim reversal features
print("\n[2.3] Linear (3 seeds, 10 dim reversal features)")
linear_dir = INTEGRATED_DIR / 'linear_regression'
for f in sorted(os.listdir(linear_dir)):
    if f.startswith('oof_scores_seed'):
        oof = np.load(linear_dir / f)
        day_oof = oof[test_last_idx]
        valid = ~np.isnan(day_oof)
        print(f"  {f}: range=[{day_oof[valid].min():+.4f}, {day_oof[valid].max():+.4f}], "
              f"mean={day_oof[valid].mean():+.4f}, std={day_oof[valid].std():.4f}")

# Linear avg
lin_avg = np.zeros((114, 300), dtype=np.float32)
for f in sorted(os.listdir(linear_dir)):
    if f.startswith('oof_scores_seed'):
        lin_avg += np.load(linear_dir / f)
lin_avg /= 3
day_lin = lin_avg[test_last_idx]
print(f"  Linear_avg       : range=[{day_lin.min():+.4f}, {day_lin.max():+.4f}], "
      f"mean={day_lin.mean():+.4f}, std={day_lin.std():.4f}")

# 2.4 显式 signals (4)
print("\n[2.4] Explicit Signals (4 显式策略 signals)")
df_full = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
df_full['日期'] = pd.to_datetime(df_full['日期'])
df_full = df_full.sort_values(['股票代码', '日期']).reset_index(drop=True)
df_full['past_ret_5d'] = (df_full['收盘'] / df_full.groupby('股票代码')['收盘'].shift(5) - 1).clip(-0.3, 0.3)

idx = pd.read_csv(INDEX_CSV, parse_dates=['date']).sort_values('date').reset_index(drop=True)
idx['_d_str'] = idx['date'].dt.strftime('%Y-%m-%d')
idx['log_ret'] = np.log(idx['close'] / idx['close'].shift(1)).fillna(0)
hs300_ret_map = dict(zip(idx['_d_str'], idx['log_ret']))
df_full['log_ret'] = np.log(df_full['收盘'] / df_full['收盘'].shift(1)).fillna(0)
df_full['hs300_ret'] = df_full['日期'].dt.strftime('%Y-%m-%d').map(hs300_ret_map).fillna(0)
df_full['beta_60'] = (df_full.groupby('股票代码', sort=False)
                      .apply(lambda g: g['log_ret'].rolling(60, min_periods=20).corr(g['hs300_ret']),
                             include_groups=False)
                      .reset_index(level=0, drop=True))

sub_626 = df_full[df_full['日期'] == '2026-06-26'].copy()
print(f"  past_ret_5d (5d momentum): range=[{sub_626['past_ret_5d'].min():+.4f}, "
      f"{sub_626['past_ret_5d'].max():+.4f}], mean={sub_626['past_ret_5d'].mean():+.4f}")
print(f"  beta_60     (60d β vs HS300): range=[{sub_626['beta_60'].min():+.4f}, "
      f"{sub_626['beta_60'].max():+.4f}], mean={sub_626['beta_60'].mean():+.4f}")
print(f"  reversal_score  = -past_ret_5d: bullish if stock dropped recently")
print(f"  momentum_score  = +past_ret_5d: bullish if stock rose recently")
print(f"  low_beta_score  = -beta_60     : bullish if stock low-β (defensive)")
print(f"  high_beta_score = +beta_60     : bullish if stock high-β (cyclical)")


# === 3. Stack 模型的 design 跟权重 ===
print("\n" + "=" * 80)
print("[3] STACK MODELS (3 regime-specific LightGBM, feature importance)")
print("=" * 80)

ensemble_dir = MODEL_INTEGRATED_DIR / 'ensemble'
STACK_FEATURES_PER_REGIME = {
    'bear':     ['tree_pred', 'linear_pred', 'macro_bias', 'reversal_score', 'low_beta_score', 'industry_id'],
    'sideways': ['tree_pred', 'linear_pred', 'macro_bias', 'macro_beta',  'diversified_score', 'industry_id'],
    'bull':     ['tree_pred', 'linear_pred', 'macro_bias', 'momentum_score', 'high_beta_score', 'industry_id'],
}
STRATEGY_MAP = {
    'bear':     'defensive: 选 low_beta + 输家反弹 (reversal)',
    'sideways': 'balanced: macro_beta + industry rotation (diversified)',
    'bull':     'momentum: 选 high_beta + 强者恒强 (momentum)',
}

for r_name in ['bear', 'sideways', 'bull']:
    pkl_p = ensemble_dir / f'stack_{r_name}.pkl'
    if not pkl_p.exists():
        print(f"\n  [{r_name}] model not found, skip")
        continue
    model = joblib.load(pkl_p)
    features = STACK_FEATURES_PER_REGIME[r_name]
    fi = model.feature_importances_
    print(f"\n  [{r_name.upper()}_STACK] strategy: {STRATEGY_MAP[r_name]}")
    print(f"    Features (6 dim): {features}")
    print(f"    n_estimators: {model.n_estimators}, max_depth: {model.max_depth}, "
          f"learning_rate: {model.learning_rate}")
    print(f"    Feature importance (gain, sorted):")
    for name, imp in sorted(zip(features, fi), key=lambda x: -x[1]):
        marker = " (strategy)" if name in ['reversal_score', 'momentum_score', 'low_beta_score',
                                            'high_beta_score', 'macro_beta', 'diversified_score'] else ""
        print(f"      {name:20s} {imp:>4d}{marker}")
    print(f"    Total: {fi.sum()}, dominant feature: {features[fi.argmax()]} "
          f"({fi.max()/fi.sum()*100:.1f}%)")


# === 4. Forecast 最终 prediction per top-5 picks ===
print("\n" + "=" * 80)
print("[4] FORECAST 6.29-7.3 TOP-5 PICKS (industry_diversified)")
print("=" * 80)
result = pd.read_csv(INTEGRATED_DIR / 'result.csv', encoding='utf-8')
print(f"\n  {result.to_string(index=False)}")
print(f"  Sum: {result['weight'].sum():.2f}, num: {len(result)}")
