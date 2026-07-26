"""Generate submission CSV.

输出: output/integrated_v1/submission_top5.csv
  - date, rank_1_stock, rank_2_stock, ..., rank_5_stock

Strategy:
  - 用 final_oof_scores.npy (test 段 114 天)
  - 每天按 score 降序, 取 top-5 (industry_diversified: 5 个不同行业)
  - 持仓期: 5 天 (T+1 ~ T+5)
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import INTEGRATED_DIR, TRAIN_CSV
from industry_mapping import load_industry_map


def main():
    base_dir = INTEGRATED_DIR

    # Load final OOF (test 段)
    final_oof = np.load(base_dir / 'ensemble' / 'final_oof_scores.npy')
    test_meta_p = base_dir / 'macro_transformer' / 'test_meta.json'
    with open(test_meta_p) as f:
        test_meta = json.load(f)
    test_dates = test_meta['dates']
    n_days, n_stocks = final_oof.shape
    print(f'[Load] final_oof: {final_oof.shape}, test_dates: {len(test_dates)}')

    # Load stock info
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig', usecols=['股票代码'])
    all_stocks = sorted(df['股票代码'].unique())
    assert n_stocks == len(all_stocks), f'mismatch: {n_stocks} vs {len(all_stocks)}'

    # Industry map
    industry_map = load_industry_map()
    industries = sorted(set(industry_map.values()))
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    stock_industry = np.array([
        industry_to_idx.get(industry_map.get(int(s), ''), 0)
        for s in all_stocks
    ])

    # Generate top-5 picks (industry_diversified: 5 个不同行业)
    rows = []
    for d_idx, d_str in enumerate(test_dates):
        scores = final_oof[d_idx]
        # 按 score 降序, greedy pick 5 个不同行业
        sorted_idx = np.argsort(-scores)  # descending
        picked = []
        picked_industries = set()
        for s_idx in sorted_idx:
            if len(picked) >= 5:
                break
            sid = stock_industry[s_idx]
            if sid not in picked_industries:
                picked.append(int(all_stocks[s_idx]))
                picked_industries.add(sid)
        # 补足 (如果行业不够 5 个)
        if len(picked) < 5:
            for s_idx in sorted_idx:
                if int(all_stocks[s_idx]) not in picked:
                    picked.append(int(all_stocks[s_idx]))
                if len(picked) >= 5:
                    break
        rows.append({
            'date': d_str,
            'rank_1': picked[0] if len(picked) > 0 else None,
            'rank_2': picked[1] if len(picked) > 1 else None,
            'rank_3': picked[2] if len(picked) > 2 else None,
            'rank_4': picked[3] if len(picked) > 3 else None,
            'rank_5': picked[4] if len(picked) > 4 else None,
        })

    sub = pd.DataFrame(rows)
    out_path = base_dir / 'submission_top5.csv'
    sub.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'[Save] {out_path}: {len(sub)} rows, {sub.shape[1]} cols')
    print(f'\n=== Top 5 picks (first 10 days) ===')
    print(sub.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
