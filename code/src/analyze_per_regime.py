"""Per-regime analysis: bull vs bear vs sideways.

分析维度:
  1. Test set performance per regime (IC / excess / win / cum)
  2. Internal LightGBM parameters (n_trees / depth / leaves)
  3. Feature importance per regime
  4. Strategy signal distributions
  5. Per-regime monthly breakdown
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import INTEGRATED_DIR, MODEL_INTEGRATED_DIR, TRAIN_CSV, INDEX_CSV
from industry_mapping import load_industry_map
from regime_classifier import RegimeClassifier

REGIME_NAMES = ['bear', 'sideways', 'bull']
COMMON_BASE = ['tree_pred', 'linear_pred', 'macro_industry_bias', 'industry_id']
STRATEGY_FEATURES = {
    0: ['reversal_score', 'low_beta_score'],
    1: ['macro_industry_beta', 'diversified_score'],
    2: ['momentum_score', 'high_beta_score'],
}
STACK_FEATURES = {r: COMMON_BASE + STRATEGY_FEATURES[r] for r in [0, 1, 2]}


def main():
    print("=" * 80)
    print("Per-Regime Analysis: bull vs bear vs sideways (Test 2026-01-01 ~ 06-26)")
    print("=" * 80)

    # Load final OOF and metadata
    final_oof = np.load(os.path.join(INTEGRATED_DIR, 'ensemble', 'final_oof_scores.npy'))
    test_meta = json.load(open(os.path.join(INTEGRATED_DIR, 'macro_transformer', 'test_meta.json'), encoding='utf-8'))
    test_dates = test_meta['dates']
    n_test = len(test_dates)
    print(f"  final_oof shape: {final_oof.shape} (n_test={n_test}, n_stocks=300)")

    # Build y_5d lookup (target)
    print("\n[1] Building y_5d lookup ...")
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['y_5d'] = ((df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)).clip(-0.3, 0.3)
    val = df[df['日期'].dt.strftime('%Y-%m-%d').isin(set(test_dates))].copy()
    val['日期_str'] = val['日期'].dt.strftime('%Y-%m-%d')
    y_lookup = {(r['日期_str'], int(r['股票代码'])): r['y_5d']
                for _, r in val[['日期_str', '股票代码', 'y_5d']].iterrows()}

    # Get test regime labels
    print("\n[2] Predicting test regime labels ...")
    rc = RegimeClassifier()
    test_regimes, _ = rc.predict(test_dates)
    n_bear = int((test_regimes == 0).sum())
    n_side = int((test_regimes == 1).sum())
    n_bull = int((test_regimes == 2).sum())
    print(f"  bear: {n_bear} days, sideways: {n_side} days, bull: {n_bull} days")

    # HS300 5d returns (for excess)
    print("\n[3] Loading HS300 5d returns ...")
    idx_df = pd.read_csv(INDEX_CSV, encoding='utf-8-sig', parse_dates=['date'])
    idx_df = idx_df.set_index('date').sort_index()
    idx_df['y_5d'] = (idx_df['close'].shift(-5) - idx_df['close'].shift(-1)) / (idx_df['close'].shift(-1) + 1e-12)
    hs300_y = {d.strftime('%Y-%m-%d'): idx_df.loc[d, 'y_5d']
               for d in pd.to_datetime(test_dates) if d in idx_df.index}

    # =====================================================================
    # A. Per-Regime Test Performance
    # =====================================================================
    print("\n" + "=" * 80)
    print("A. Per-Regime Test Performance (final OOF, top-5 industry_diversified, 5d horizon)")
    print("=" * 80)

    industry_map = load_industry_map()
    n_stocks = final_oof.shape[1]
    all_stocks = sorted(df['股票代码'].unique().tolist())
    stock_to_idx = {s: i for i, s in enumerate(all_stocks)}

    per_regime_metrics = {}
    for r, name in enumerate(REGIME_NAMES):
        days = [i for i, rr in enumerate(test_regimes) if rr == r]
        if not days:
            per_regime_metrics[name] = None
            continue

        abs_returns, excess_returns = [], []
        ics = []
        for di in days:
            d = test_dates[di]
            scores = final_oof[di]
            y_vec = np.array([y_lookup.get((d, s), np.nan) for s in all_stocks])
            mask = ~np.isnan(y_vec)
            if mask.sum() < 5:
                continue
            ics.append(spearmanr(scores[mask], y_vec[mask]).correlation)

            # Top-5 industry-diversified selection
            order = np.argsort(-scores)
            picked, ind_seen = [], set()
            for idx in order:
                if len(picked) == 5:
                    break
                ind = industry_map.get(all_stocks[idx])
                if ind in ind_seen:
                    continue
                ind_seen.add(ind)
                picked.append(idx)
            weights = np.zeros(n_stocks)
            for idx in picked:
                weights[idx] = 0.2
            port_ret = float(np.nansum(weights * y_vec))
            abs_returns.append(port_ret)
            excess_returns.append(port_ret - hs300_y.get(d, 0))

        cum = np.prod(1 + np.array(excess_returns)) - 1
        per_regime_metrics[name] = {
            'n_days': len(days),
            'ic_mean': float(np.mean(ics)),
            'ic_std': float(np.std(ics)),
            'abs_5d': float(np.mean(abs_returns)),
            'excess_5d': float(np.mean(excess_returns)),
            'win_rate': float(np.mean(np.array(excess_returns) > 0)),
            'cum_excess': float(cum),
        }

    print(f"\n  {'regime':<10} {'n_days':>7} {'IC':>9} {'IC_std':>9} {'abs_5d':>9} {'excess_5d':>10} {'win%':>7} {'cum_excess':>12}")
    for name in REGIME_NAMES:
        m = per_regime_metrics.get(name)
        if m is None:
            print(f"  {name:<10} {'-':>7}")
            continue
        print(f"  {name:<10} {m['n_days']:>7} {m['ic_mean']:>+9.4f} {m['ic_std']:>9.4f} "
              f"{m['abs_5d']*100:>+8.3f}% {m['excess_5d']*100:>+9.3f}% {m['win_rate']*100:>6.1f}% {m['cum_excess']*100:>+11.2f}%")

    # =====================================================================
    # B. Internal LightGBM Parameters
    # =====================================================================
    print("\n" + "=" * 80)
    print("B. Internal LightGBM Parameters (per regime stack model)")
    print("=" * 80)

    print(f"\n  {'regime':<10} {'n_features':>10} {'n_trees':>8} {'max_depth':>10} {'num_leaves':>11} {'learning_rate':>14}")
    for r, name in enumerate(REGIME_NAMES):
        m = joblib.load(os.path.join(MODEL_INTEGRATED_DIR, 'ensemble', f'stack_{name}.pkl'))
        booster = m.booster_
        n_trees = booster.num_trees()
        try:
            max_depth = int(booster.params.get('max_depth', 0) or 0)
        except Exception:
            max_depth = 0
        try:
            num_leaves = int(booster.params.get('num_leaves', 0) or 0)
        except Exception:
            num_leaves = 0
        try:
            lr = float(booster.params.get('learning_rate', 0) or 0)
        except Exception:
            lr = 0.0
        n_feat = len(STACK_FEATURES[r])
        print(f"  {name:<10} {n_feat:>10} {n_trees:>8} {max_depth:>10} {num_leaves:>11} {lr:>14.4f}")

    # =====================================================================
    # C. Feature Importance per Regime
    # =====================================================================
    print("\n" + "=" * 80)
    print("C. Feature Importance per Regime (gain)")
    print("=" * 80)

    fi_per_regime = {}
    for r, name in enumerate(REGIME_NAMES):
        m = joblib.load(os.path.join(MODEL_INTEGRATED_DIR, 'ensemble', f'stack_{name}.pkl'))
        fi = m.feature_importances_
        fi_per_regime[name] = dict(zip(STACK_FEATURES[r], fi))

    all_features = COMMON_BASE + ['reversal_score', 'low_beta_score', 'macro_industry_beta',
                                  'diversified_score', 'momentum_score', 'high_beta_score']
    print(f"\n  {'feature':<22} {'bear':>8} {'sideways':>10} {'bull':>8}")
    for f in all_features:
        b = fi_per_regime['bear'].get(f, 0)
        s = fi_per_regime['sideways'].get(f, 0)
        u = fi_per_regime['bull'].get(f, 0)
        marker = ''
        if f in STRATEGY_FEATURES[0]:
            marker = ' [bear]'
        elif f in STRATEGY_FEATURES[1]:
            marker = ' [side]'
        elif f in STRATEGY_FEATURES[2]:
            marker = ' [bull]'
        print(f"  {f+marker:<22} {b:>8.1f} {s:>10.1f} {u:>8.1f}")

    # =====================================================================
    # D. Strategy Signal Distributions
    # =====================================================================
    print("\n" + "=" * 80)
    print("D. Strategy Signal Distributions (per regime strategy features)")
    print("=" * 80)

    # Load val + test signals
    print("\n  Building explicit signals ...")
    val_meta = json.load(open(os.path.join(INTEGRATED_DIR, 'tree', 'val_meta.json'), encoding='utf-8'))
    val_dates = val_meta['dates']
    val_regimes, _ = rc.predict(val_dates)

    # Reconstruct signals (same logic as build_explicit_signals)
    sig = df.copy()
    sig = sig.sort_values(['股票代码', '日期']).reset_index(drop=True)
    sig['日期_str'] = sig['日期'].dt.strftime('%Y-%m-%d')
    for w in [5, 10, 20, 60]:
        sig[f'past_ret_{w}d'] = sig.groupby('股票代码')['收盘'].pct_change(w)

    # 60d rolling beta vs HS300 (correct: corr(stock_ret, hs300_ret))
    idx_series = idx_df['close'].pct_change().to_dict()
    sig['idx_ret'] = sig['日期'].map(idx_series)
    sig['log_ret'] = np.log(sig['收盘'] / sig.groupby('股票代码')['收盘'].shift(1)).fillna(0)
    betas = []
    for sid, grp in sig.groupby('股票代码'):
        beta = grp['log_ret'].rolling(60, min_periods=20).corr(grp['idx_ret'])
        betas.append(pd.DataFrame({'股票代码': sid, '日期': grp['日期'], 'beta_60': beta}))
    beta_df = pd.concat(betas, ignore_index=True)
    sig = sig.merge(beta_df[['股票代码', '日期', 'beta_60']], on=['股票代码', '日期'], how='left')

    # Per-regime strategy signal stats (test only)
    print(f"\n  Test set (n={n_test} days, 300 stocks/day):")
    print(f"  {'signal':<22} {'bear':>22} {'sideways':>22} {'bull':>22}")
    print(f"  {'':<22} {'mean / std':>22} {'mean / std':>22} {'mean / std':>22}")

    test_dates_set = set(test_dates)
    test_sig = sig[sig['日期_str'].isin(test_dates_set)].copy()

    for r, name in enumerate(REGIME_NAMES):
        days = [d for d, rr in zip(test_dates, test_regimes) if rr == r]
        if not days:
            continue
        sub = test_sig[test_sig['日期_str'].isin(days)]
        if r == 0:  # bear: reversal + low_beta
            print(f"  {'reversal_score':<22} {(sub['past_ret_5d']*-1).mean():>+8.4f} / {(sub['past_ret_5d']*-1).std():>8.4f}  {'(-past_ret_5d)':>4}")
            print(f"  {'low_beta_score':<22} {(sub['beta_60']*-1).mean():>+8.4f} / {(sub['beta_60']*-1).std():>8.4f}  {'(-beta_60)':>4}")
        elif r == 2:  # bull: momentum + high_beta
            print(f"  {'momentum_score':<22} {sub['past_ret_5d'].mean():>+8.4f} / {sub['past_ret_5d'].std():>8.4f}  {'(+past_ret_5d)':>4}")
            print(f"  {'high_beta_score':<22} {sub['beta_60'].mean():>+8.4f} / {sub['beta_60'].std():>8.4f}  {'(+beta_60)':>4}")
        elif r == 1:
            print(f"  (sideways: macro_industry_beta + diversified_score 来自 macro_pred, 不依赖 stock-level 分布)")

    # =====================================================================
    # E. Per-Regime Monthly Performance
    # =====================================================================
    print("\n" + "=" * 80)
    print("E. Per-Regime Monthly Breakdown")
    print("=" * 80)

    monthly = defaultdict(lambda: defaultdict(list))
    for di, d in enumerate(test_dates):
        rr = test_regimes[di]
        name = REGIME_NAMES[rr]
        month = d[:7]
        scores = final_oof[di]
        y_vec = np.array([y_lookup.get((d, s), np.nan) for s in all_stocks])
        mask = ~np.isnan(y_vec)
        if mask.sum() < 5:
            continue
        order = np.argsort(-scores)
        picked, ind_seen = [], set()
        for idx in order:
            if len(picked) == 5:
                break
            ind = industry_map.get(all_stocks[idx])
            if ind in ind_seen:
                continue
            ind_seen.add(ind)
            picked.append(idx)
        weights = np.zeros(n_stocks)
        for idx in picked:
            weights[idx] = 0.2
        port_ret = float(np.nansum(weights * y_vec))
        excess = port_ret - hs300_y.get(d, 0)
        monthly[month][name].append(excess)

    months = sorted(monthly.keys())
    print(f"\n  {'month':<10}", end='')
    for name in REGIME_NAMES:
        print(f"  {name+' excess_5d':>16}", end='')
    print()
    for m in months:
        print(f"  {m:<10}", end='')
        for name in REGIME_NAMES:
            vals = monthly[m].get(name, [])
            if not vals:
                print(f"  {'-':>16}", end='')
            else:
                print(f"  {np.mean(vals)*100:>+15.3f}%", end='')
        print()

    # =====================================================================
    # F. Summary table
    # =====================================================================
    print("\n" + "=" * 80)
    print("F. Summary: 3 stack models side-by-side")
    print("=" * 80)

    print(f"\n  {'aspect':<28} {'bear':>14} {'sideways':>14} {'bull':>14}")
    print(f"  {'strategy':<28} {'reversal+lowβ':>14} {'macroβ+div':>14} {'mom+highβ':>14}")
    print(f"  {'n_test_days':<28} {n_bear:>14} {n_side:>14} {n_bull:>14}")
    print(f"  {'IC mean':<28} {per_regime_metrics['bear']['ic_mean']:>+14.4f} {per_regime_metrics['sideways']['ic_mean']:>+14.4f} {per_regime_metrics['bull']['ic_mean']:>+14.4f}")
    print(f"  {'excess 5d':<28} {per_regime_metrics['bear']['excess_5d']*100:>+13.3f}% {per_regime_metrics['sideways']['excess_5d']*100:>+13.3f}% {per_regime_metrics['bull']['excess_5d']*100:>+13.3f}%")
    print(f"  {'cum_excess':<28} {per_regime_metrics['bear']['cum_excess']*100:>+13.2f}% {per_regime_metrics['sideways']['cum_excess']*100:>+13.2f}% {per_regime_metrics['bull']['cum_excess']*100:>+13.2f}%")
    print(f"  {'win rate':<28} {per_regime_metrics['bear']['win_rate']*100:>+13.1f}% {per_regime_metrics['sideways']['win_rate']*100:>+13.1f}% {per_regime_metrics['bull']['win_rate']*100:>+13.1f}%")
    print("\n[Done] Per-regime analysis complete")


if __name__ == '__main__':
    main()
