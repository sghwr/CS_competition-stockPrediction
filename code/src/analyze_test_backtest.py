"""Test 段 (6 月) 回测深度分析:
1. 4 metrics 对比 (final / transformer / tree / linear)
2. regime-specific performance
3. 月度 performance
4. Cumulative wealth curve
5. Top-5 picks vs actual winners
6. IC analysis
7. Industry allocation
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from collections import Counter, defaultdict
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import INTEGRATED_DIR, MODEL_INTEGRATED_DIR, TRAIN_CSV, INDEX_CSV
from regime_classifier import RegimeClassifier
from industry_mapping import load_industry_map


def main():
    print("=" * 80)
    print("Test 段 (6 月, 2026-01-05 ~ 2026-06-26) Backtest 深度分析")
    print("=" * 80)

    # === Load data ===
    print("\n[Load] loading data ...")
    with open(INTEGRATED_DIR / 'macro_transformer' / 'test_meta.json', encoding='utf-8') as f:
        test_meta = json.load(f)
    test_dates = test_meta['dates']
    n_days = len(test_dates)
    print(f"  test dates: {test_dates[0]} ~ {test_dates[-1]} ({n_days} days)")

    # 4 OOF arrays
    final_oof = np.load(INTEGRATED_DIR / 'ensemble' / 'final_oof_scores.npy')
    macro_npz = np.load(INTEGRATED_DIR / 'macro_transformer' / 'macro_pred_avg.npz', allow_pickle=True)
    macro_all = {
        'dates': list(macro_npz['dates']),
        'industry_bias': macro_npz['industry_bias'],
        'industry_beta': macro_npz['industry_beta'],
        'industries': list(macro_npz['industries']),
    }
    industries = macro_all['industries']
    industry_map = load_industry_map()
    all_stocks = sorted(pd.read_csv(TRAIN_CSV, encoding='utf-8-sig',
                                     usecols=['股票代码'])['股票代码'].unique())
    stockid2idx = {int(s): i for i, s in enumerate(all_stocks)}
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    stock_industry_idx = np.array([
        industry_to_idx.get(industry_map.get(int(s), ''), 0) for s in all_stocks
    ])

    # Per-day 行业 bias
    test_idx_in_macro = [macro_all['dates'].index(d) for d in test_dates]
    industry_bias_per_day = macro_all['industry_bias'][test_idx_in_macro]  # (n_days, 11)

    # Tree/Linear OOF
    tree_bull = np.load(INTEGRATED_DIR / 'tree' / 'oof_scores_bull.npy')
    tree_side = np.load(INTEGRATED_DIR / 'tree' / 'oof_scores_sideways.npy')
    tree_bear = np.load(INTEGRATED_DIR / 'tree' / 'oof_scores_bear.npy')
    lin_files = [f for f in os.listdir(INTEGRATED_DIR / 'linear_regression') if f.startswith('oof_scores_seed')]
    lin_avg = np.zeros_like(tree_bull)
    for f in lin_files:
        lin_avg += np.load(INTEGRATED_DIR / 'linear_regression' / f)
    lin_avg /= len(lin_files)

    # Stock prices
    print("[Load] stock prices ...")
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    df['_d_str'] = df['日期'].dt.strftime('%Y-%m-%d')
    # pivot opens: (n_days, 300)
    open_pivot = np.full((n_days, len(all_stocks)), np.nan, dtype=np.float32)
    sub = df[df['_d_str'].isin(test_dates)][['股票代码', '_d_str', '开盘']].copy()
    for _, row in sub.iterrows():
        d_idx = test_dates.index(row['_d_str'])
        s_idx = stockid2idx.get(int(row['股票代码']), None)
        if s_idx is not None and pd.notna(row['开盘']):
            open_pivot[d_idx, s_idx] = float(row['开盘'])

    # Compute 5d forward return
    forward_5d = np.full((n_days, len(all_stocks)), np.nan, dtype=np.float32)
    for d_idx in range(n_days - 5):
        # T+1 open
        t1 = open_pivot[d_idx + 1]
        t5 = open_pivot[d_idx + 5]
        valid = ~np.isnan(t1) & ~np.isnan(t5) & (t1 > 0)
        forward_5d[d_idx, valid] = (t5[valid] - t1[valid]) / t1[valid]
    valid_days = ~np.isnan(forward_5d).all(axis=1)
    valid_days[-5:] = False
    valid_day_idx = np.where(valid_days)[0]
    print(f"  valid days (5d forward 可算): {len(valid_day_idx)}")

    # HS300
    idx = pd.read_csv(INDEX_CSV, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    idx['_d_str'] = idx['date'].dt.strftime('%Y-%m-%d')
    hs300_open = np.full(n_days, np.nan, dtype=np.float32)
    for d_idx, d in enumerate(test_dates):
        row = idx[idx['_d_str'] == d]
        if len(row) > 0:
            hs300_open[d_idx] = float(row['open'].iloc[0])
    hs300_5d = np.full(n_days, np.nan)
    for d_idx in range(n_days - 5):
        t1 = hs300_open[d_idx + 1]
        t5 = hs300_open[d_idx + 5]
        if t1 > 0 and not np.isnan(t1) and not np.isnan(t5):
            hs300_5d[d_idx] = (t5 - t1) / t1
    print(f"  HS300 5d mean return: {np.nanmean(hs300_5d[valid_day_idx])*100:+.4f}%/day")

    # Regime per day
    rc = RegimeClassifier()
    test_regimes, _ = rc.predict(test_dates)
    print(f"  test regime: bear={sum(test_regimes==0)}, side={sum(test_regimes==1)}, bull={sum(test_regimes==2)}")

    # === 1. 4 OOF 总体表现 ===
    print("\n" + "=" * 80)
    print("[1] 4 OOF 总表现对比 (industry_diversified, top-5, 5d horizon)")
    print("=" * 80)
    print(f"  {'OOF':<10s}  {'abs%/d':>10s}  {'excess%/d':>10s}  {'top5_ratio':>10s}  {'NDCG@5':>8s}  {'win%':>6s}  {'cum_abs':>10s}")
    for name, oof in [('final', final_oof), ('tree', tree_side), ('linear', lin_avg)]:
        # Daily 5 picks
        daily_picks = []
        for d_idx in valid_day_idx:
            scores = oof[d_idx]
            # industry_diversified
            sorted_idx = np.argsort(-scores)
            picked, picked_ind = [], set()
            for s_idx in sorted_idx:
                ind = stock_industry_idx[s_idx]
                if ind not in picked_ind:
                    picked.append(s_idx)
                    picked_ind.add(ind)
                if len(picked) >= 5:
                    break
            daily_picks.append(picked)
        daily_picks = np.array(daily_picks)  # (n_valid, 5)

        # Returns
        rets = []
        for i, d_idx in enumerate(valid_day_idx):
            r = forward_5d[d_idx, daily_picks[i]].mean()
            rets.append(r)
        rets = np.array(rets)
        hs = hs300_5d[valid_day_idx]
        excess = rets - hs
        win = (excess > 0).mean()

        # NDCG@5
        ndcg_list = []
        for i, d_idx in enumerate(valid_day_idx):
            actual = forward_5d[d_idx]
            valid = ~np.isnan(actual)
            if valid.sum() < 5:
                continue
            actual_v = actual.copy()
            actual_v[~valid] = -np.inf
            actual_top5 = np.argsort(-actual_v)[:5]
            picked_top5 = daily_picks[i]
            ideal = np.sort(actual_v[actual_top5])[::-1]
            dcg = sum((2**actual_v[p] - 1) / np.log2(k+2) for k, p in enumerate(picked_top5))
            idcg = sum((2**ideal[k] - 1) / np.log2(k+2) for k in range(5))
            if idcg > 0:
                ndcg_list.append(dcg / idcg)
        ndcg = np.mean(ndcg_list) if ndcg_list else 0

        # top5 ratio
        top5_avg = []
        for i, d_idx in enumerate(valid_day_idx):
            actual = forward_5d[d_idx]
            valid = ~np.isnan(actual)
            if valid.sum() < 5:
                continue
            actual_v = actual.copy()
            actual_v[~valid] = -np.inf
            actual_top5 = np.argsort(-actual_v)[:5]
            top5_avg.append(actual[actual_top5].mean())
        top5_avg = np.array(top5_avg)
        ratio = (rets / (top5_avg + 1e-12)).mean()

        # cumulative
        cum_abs = (1 + rets).prod() - 1
        cum_excess = (1 + excess).prod() - 1

        print(f"  {name:<10s}  {rets.mean()*100:>+10.4f}  {excess.mean()*100:>+10.4f}  "
              f"{ratio:>+10.4f}  {ndcg:>8.4f}  {win*100:>5.1f}%  {cum_abs*100:>+10.2f}")

    # === 2. regime-specific 表现 (final OOF) ===
    print("\n" + "=" * 80)
    print("[2] Regime-Specific 表现 (final OOF, 6 月 test 段)")
    print("=" * 80)
    print(f"  {'Regime':<10s}  {'#days':>6s}  {'abs%/d':>10s}  {'excess%/d':>10s}  {'win%':>6s}  {'IC':>8s}")
    for r in [0, 1, 2]:
        r_name = ['bear', 'sideways', 'bull'][r]
        days = [i for i in valid_day_idx if int(test_regimes[i]) == r]
        if len(days) == 0:
            continue
        rets_r = []
        for d_idx in days:
            scores = final_oof[d_idx]
            sorted_idx = np.argsort(-scores)
            picked, picked_ind = [], set()
            for s_idx in sorted_idx:
                ind = stock_industry_idx[s_idx]
                if ind not in picked_ind:
                    picked.append(s_idx)
                    picked_ind.add(ind)
                if len(picked) >= 5:
                    break
            r_ = forward_5d[d_idx, picked].mean()
            rets_r.append(r_)
        rets_r = np.array(rets_r)
        hs_r = hs300_5d[days]
        excess_r = rets_r - hs_r

        # IC
        ics = []
        for d_idx in days:
            scores = final_oof[d_idx]
            actual = forward_5d[d_idx]
            valid = ~np.isnan(scores) & ~np.isnan(actual)
            if valid.sum() > 5:
                ic, _ = spearmanr(scores[valid], actual[valid])
                ics.append(ic)
        ic_mean = np.mean(ics) if ics else 0

        print(f"  {r_name:<10s}  {len(days):>6d}  {rets_r.mean()*100:>+10.4f}  "
              f"{excess_r.mean()*100:>+10.4f}  {(excess_r>0).mean()*100:>5.1f}%  {ic_mean:>+8.4f}")

    # === 3. 月度表现 ===
    print("\n" + "=" * 80)
    print("[3] 月度表现 (final OOF)")
    print("=" * 80)
    print(f"  {'Month':<10s}  {'#days':>6s}  {'abs%/d':>10s}  {'excess%/d':>10s}  {'win%':>6s}")
    monthly = defaultdict(list)
    for i, d_idx in enumerate(valid_day_idx):
        m = test_dates[d_idx][:7]
        scores = final_oof[d_idx]
        sorted_idx = np.argsort(-scores)
        picked, picked_ind = [], set()
        for s_idx in sorted_idx:
            ind = stock_industry_idx[s_idx]
            if ind not in picked_ind:
                picked.append(s_idx)
                picked_ind.add(ind)
            if len(picked) >= 5:
                break
        r_ = forward_5d[d_idx, picked].mean()
        monthly[m].append((r_, hs300_5d[d_idx]))
    for m in sorted(monthly.keys()):
        rets_m = np.array([x[0] for x in monthly[m]])
        hs_m = np.array([x[1] for x in monthly[m]])
        excess_m = rets_m - hs_m
        print(f"  {m:<10s}  {len(rets_m):>6d}  {rets_m.mean()*100:>+10.4f}  "
              f"{excess_m.mean()*100:>+10.4f}  {(excess_m>0).mean()*100:>5.1f}%")

    # === 4. IC analysis (per regime) ===
    print("\n" + "=" * 80)
    print("[4] IC Analysis (Spearman, daily)")
    print("=" * 80)
    print(f"  {'OOF':<10s}  {'overall':>10s}  {'bear':>10s}  {'side':>10s}  {'bull':>10s}")
    for name, oof in [('final', final_oof), ('tree', tree_side), ('linear', lin_avg),
                      ('tree_bull', tree_bull), ('tree_bear', tree_bear)]:
        ics_all, ics_b, ics_s, ics_u = [], [], [], []
        for d_idx in valid_day_idx:
            scores = oof[d_idx]
            actual = forward_5d[d_idx]
            valid = ~np.isnan(scores) & ~np.isnan(actual)
            if valid.sum() > 5:
                ic, _ = spearmanr(scores[valid], actual[valid])
                ics_all.append(ic)
                r = int(test_regimes[d_idx])
                if r == 0:
                    ics_b.append(ic)
                elif r == 1:
                    ics_s.append(ic)
                else:
                    ics_u.append(ic)
        print(f"  {name:<10s}  {np.mean(ics_all):>+10.4f}  {np.mean(ics_b):>+10.4f}  "
              f"{np.mean(ics_s):>+10.4f}  {np.mean(ics_u):>+10.4f}")

    # === 5. 行业分配 (top-5 picks per day) ===
    print("\n" + "=" * 80)
    print("[5] Industry Allocation (109 days, top-5 industry_diversified picks)")
    print("=" * 80)
    ind_count = Counter()
    for d_idx in valid_day_idx:
        scores = final_oof[d_idx]
        sorted_idx = np.argsort(-scores)
        picked, picked_ind = [], set()
        for s_idx in sorted_idx:
            ind = stock_industry_idx[s_idx]
            if ind not in picked_ind:
                picked.append(s_idx)
                picked_ind.add(ind)
            if len(picked) >= 5:
                break
        for s_idx in picked:
            ind_count[industries[stock_industry_idx[s_idx]]] += 1
    total_picks = sum(ind_count.values())
    print(f"  Total picks: {total_picks} ({len(valid_day_idx)} days × 5)")
    for ind in sorted(ind_count.keys(), key=lambda x: -ind_count[x]):
        c = ind_count[ind]
        print(f"    {ind:10s}  {c:>4d} ({c/total_picks*100:>5.1f}%)")

    # === 6. 累计 wealth curve (10 段) ===
    print("\n" + "=" * 80)
    print("[6] Cumulative Wealth Curve (109 days, final OOF)")
    print("=" * 80)
    daily_picks = []
    for d_idx in valid_day_idx:
        scores = final_oof[d_idx]
        sorted_idx = np.argsort(-scores)
        picked, picked_ind = [], set()
        for s_idx in sorted_idx:
            ind = stock_industry_idx[s_idx]
            if ind not in picked_ind:
                picked.append(s_idx)
                picked_ind.add(ind)
            if len(picked) >= 5:
                break
        daily_picks.append(picked)
    rets = []
    for i, d_idx in enumerate(valid_day_idx):
        r = forward_5d[d_idx, daily_picks[i]].mean()
        rets.append(r)
    rets = np.array(rets)
    hs = hs300_5d[valid_day_idx]
    cum_abs = (1 + rets).cumprod()
    cum_hs = (1 + hs).cumprod()
    cum_excess = cum_abs / cum_hs  # ratio

    # Split into 10 段
    n = len(cum_abs)
    segments = np.array_split(np.arange(n), 10)
    print(f"  {'Segment':<10s}  {'cum_abs':>10s}  {'cum_hs':>10s}  {'excess':>10s}")
    for i, seg in enumerate(segments):
        s, e = seg[0], seg[-1] + 1
        print(f"  Day {s+1:>3d}-{e:<3d}  {cum_abs[e-1]:>10.4f}  {cum_hs[e-1]:>10.4f}  "
              f"{(cum_abs[e-1]/cum_hs[e-1]-1)*100:>+9.2f}%")

    # === 7. Tree OOF 6.22-6.26 (新加的) ===
    print("\n" + "=" * 80)
    print("[7] Tree/Linear OOF 6.22-6.26 完整性检查 (re-extract 后)")
    print("=" * 80)
    for r_name in ['bear', 'sideways', 'bull']:
        oof = np.load(INTEGRATED_DIR / 'tree' / f'oof_scores_{r_name}.npy')
        last5_valid = sum((~np.isnan(oof[i])).sum() for i in range(-5, 0))
        print(f"  Tree_{r_name:8s}: last 5 days total valid = {last5_valid}/1500")
    for f in lin_files:
        oof = np.load(INTEGRATED_DIR / 'linear_regression' / f)
        last5_valid = sum((~np.isnan(oof[i])).sum() for i in range(-5, 0))
        print(f"  {f}: last 5 days total valid = {last5_valid}/1500")


if __name__ == '__main__':
    main()
