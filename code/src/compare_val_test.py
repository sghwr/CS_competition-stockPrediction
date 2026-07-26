"""
对比验证集 (2025) vs 测试集 (2026) 的 backtest 结果
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from backtest import backtest_one, build_price_lookups
from paths import PROJECT_ROOT, INTEGRATED_DIR, TRAIN_CSV, INDEX_CSV


def load_val_data(base_dir):
    """Load val OOF (2025) from val_meta and val_pred."""
    base_dir = Path(base_dir)
    with open(base_dir / 'macro_transformer' / 'val_meta.json') as f:
        meta = json.load(f)
    val_dates = meta['dates']
    print(f"  val dates: {val_dates[0]} ~ {val_dates[-1]} ({len(val_dates)} days)")

    final_oof = np.load(base_dir / 'ensemble' / 'val_pred.npy')

    from industry_mapping import load_industry_map
    macro_npz = base_dir / 'macro_transformer' / 'macro_pred_seed42.npz'
    if not macro_npz.exists():
        macro_npz = base_dir / 'macro_transformer' / 'macro_pred_avg.npz'
    with np.load(macro_npz, allow_pickle=True) as z:
        t_dates_all = list(z['dates'])
        ib_all = z['industry_bias']
        industries = list(z['industries'])
    t_date_to_idx = {d: i for i, d in enumerate(t_dates_all)}
    val_indices = [t_date_to_idx[d] for d in val_dates if d in t_date_to_idx]
    ib = ib_all[val_indices]
    industry_map = load_industry_map()
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    all_stocks_full = sorted(pd.read_csv(TRAIN_CSV, encoding='utf-8-sig', usecols=['股票代码'])['股票代码'].unique())
    stock_ind = np.array([industry_to_idx.get(industry_map.get(int(s), ''), 0) for s in all_stocks_full])
    transformer_oof = ib[:, stock_ind]

    tree_oof = np.load(base_dir / 'tree' / 'val_oof.npy')
    linear_oof = np.load(base_dir / 'linear_regression' / 'val_oof_seed42.npy')

    print(f"  final: {final_oof.shape}, transformer: {transformer_oof.shape}, "
          f"tree: {tree_oof.shape}, linear: {linear_oof.shape}")
    return final_oof, transformer_oof, tree_oof, linear_oof, val_dates


def load_test_data(base_dir):
    """Load test OOF (2026) from test_meta."""
    base_dir = Path(base_dir)
    with open(base_dir / 'macro_transformer' / 'test_meta.json') as f:
        meta = json.load(f)
    test_dates = meta['dates']
    print(f"  test dates: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)} days)")

    final_oof = np.load(base_dir / 'ensemble' / 'final_oof_scores.npy')

    from industry_mapping import load_industry_map
    macro_npz = base_dir / 'macro_transformer' / 'macro_pred_seed42.npz'
    if not macro_npz.exists():
        macro_npz = base_dir / 'macro_transformer' / 'macro_pred_avg.npz'
    with np.load(macro_npz, allow_pickle=True) as z:
        t_dates_all = list(z['dates'])
        ib_all = z['industry_bias']
        industries = list(z['industries'])
    t_date_to_idx = {d: i for i, d in enumerate(t_dates_all)}
    test_indices = [t_date_to_idx[d] for d in test_dates if d in t_date_to_idx]
    ib = ib_all[test_indices]
    industry_map = load_industry_map()
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    all_stocks_full = sorted(pd.read_csv(TRAIN_CSV, encoding='utf-8-sig', usecols=['股票代码'])['股票代码'].unique())
    stock_ind = np.array([industry_to_idx.get(industry_map.get(int(s), ''), 0) for s in all_stocks_full])
    transformer_oof = ib[:, stock_ind]

    tree_oof = np.load(base_dir / 'tree' / 'oof_scores.npy')
    linear_oof = np.load(base_dir / 'linear_regression' / 'oof_scores_seed42.npy')

    print(f"  final: {final_oof.shape}, transformer: {transformer_oof.shape}, "
          f"tree: {tree_oof.shape}, linear: {linear_oof.shape}")
    return final_oof, transformer_oof, tree_oof, linear_oof, test_dates


def run_one(oof_dict, dates, prices, hs300, selection='industry_diversified', label=''):
    all_stocks = sorted(prices['stock'].unique())
    price_open, hs300_open = build_price_lookups(prices, hs300)
    results = {}
    for name, oof in oof_dict.items():
        summary, _ = backtest_one(
            oof, dates, all_stocks, price_open, hs300_open,
            top_k=5, horizon=5, name=f'{name}_{label}',
            selection_mode=selection,
        )
        if summary is not None:
            results[name] = summary
    return results


def main():
    base_dir = INTEGRATED_DIR

    print("=" * 70)
    print("Loading prices & HS300 ...")
    prices = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    prices = prices.rename(columns={'日期': 'date', '股票代码': 'stock'})
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.sort_values(['stock', 'date']).reset_index(drop=True)
    print(f"  prices: {len(prices):,} rows, {prices['stock'].nunique()} stocks, "
          f"{prices['date'].min().date()} ~ {prices['date'].max().date()}")

    hs300_all = pd.read_csv(INDEX_CSV)
    hs300_all['date'] = pd.to_datetime(hs300_all['date'])
    hs300_all = hs300_all.sort_values('date').reset_index(drop=True)
    print(f"  HS300: {len(hs300_all)} days")

    print("\n" + "=" * 70)
    print("Loading VAL data (2025) ...")
    val_final, val_t, val_g, val_l, val_dates = load_val_data(base_dir)
    val_oofs = {'final': val_final, 'transformer': val_t, 'tree': val_g, 'linear_reversal': val_l}

    print("\n" + "=" * 70)
    print("Loading TEST data (2026) ...")
    test_final, test_t, test_g, test_l, test_dates = load_test_data(base_dir)
    test_oofs = {'final': test_final, 'transformer': test_t, 'tree': test_g, 'linear_reversal': test_l}

    print("\n" + "=" * 70)
    print("Running backtest on VAL (2025) ...")
    val_results = run_one(val_oofs, val_dates, prices, hs300_all, label='val')

    print("\n" + "=" * 70)
    print("Running backtest on TEST (2026) ...")
    test_results = run_one(test_oofs, test_dates, prices, hs300_all, label='test')

    print("\n" + "=" * 70)
    print("Comparison: VAL (2025) vs TEST (2026)")
    print("=" * 70)
    metrics = ['abs_return', 'excess_return', 'top5_ratio', 'ndcg5', 'cum_excess', 'excess_win_rate']
    header = f"{'Branch':<20} {'Period':<8}" + ''.join(f'{m:>14}' for m in metrics)
    print(header)
    print('-' * len(header))

    for name in ['final', 'transformer', 'tree', 'linear_reversal']:
        vr = val_results.get(name)
        tr = test_results.get(name)
        if vr:
            vals = [f'{vr.get(m, 0)*100:>+12.4f}%' if 'win' not in m and 'ratio' not in m and 'ndcg' not in m
                    else f'{vr.get(m, 0):>+13.4f}' for m in metrics]
            print(f'{name:<20} {"VAL":<8}' + ''.join(f'{v:>14}' for v in vals))
        if tr:
            vals = [f'{tr.get(m, 0)*100:>+12.4f}%' if 'win' not in m and 'ratio' not in m and 'ndcg' not in m
                    else f'{tr.get(m, 0):>+13.4f}' for m in metrics]
            print(f'{name:<20} {"TEST":<8}' + ''.join(f'{v:>14}' for v in vals))
        print()

    # save
    out = {'val': {k: {mk: v for mk, v in r.items()} for k, r in val_results.items()},
           'test': {k: {mk: v for mk, v in r.items()} for k, r in test_results.items()}}
    with open(INTEGRATED_DIR / 'backtest' / 'val_test_comparison.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[Save] {INTEGRATED_DIR / 'backtest' / 'val_test_comparison.json'}")


if __name__ == '__main__':
    main()
