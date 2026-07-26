"""Wrapper: 用现成 Tree/Linear model 重新 inference 6.22-6.26
不需 label (forward 5d), 只用 features 来预测。
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from utils import (
    engineer_features_158plus39,
    enrich_window_factors,
    TIER1_LAST_ONLY,
    TIER2_ENRICHED,
    TIER3_INDUSTRY,
)
from industry_mapping import load_industry_map
from paths import INTEGRATED_DIR, MODEL_INTEGRATED_DIR, TRAIN_CSV


def build_features_for_dates(target_dates_str, scaler_path, feature_names_path):
    """Build features for target dates only (no label needed).
    target_dates_str: list of 'YYYY-MM-DD' strings
    """
    print(f"\n[Data] Reading stock data for {target_dates_str}...")
    df = pd.read_csv(TRAIN_CSV)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # Need 60d context: from earlier (use 2025-12-01) to target
    target_start = pd.Timestamp('2025-12-01')  # well before 60d window
    target_end = pd.Timestamp(target_dates_str[-1])

    context_df = df[(df['日期'] >= target_start) & (df['日期'] <= target_end)].copy()
    print(f"  context: {context_df['日期'].min().date()} to {context_df['日期'].max().date()}, "
          f"{len(context_df)} rows")

    # Engineer features
    print(f"\n[Engineer] Running engineer_features_158plus39 on context...")
    all_groups = [g for _, g in context_df.groupby('股票代码', sort=False)]
    processed_list = [None] * len(all_groups)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(engineer_features_158plus39, g): i for i, g in enumerate(all_groups)}
        for fut in as_completed(futures):
            processed_list[futures[fut]] = fut.result()
    processed = pd.concat(processed_list).reset_index(drop=True)
    print(f"  done: {len(processed)} rows, {len(processed.columns)} cols")

    # Clean inf/nan
    cols_to_clean = TIER1_LAST_ONLY + TIER2_ENRICHED
    cols_present = [c for c in cols_to_clean if c in processed.columns]
    for c in cols_present:
        processed[c] = processed[c].replace([np.inf, -np.inf], 0).fillna(0)

    # Add industry_id
    industry_map = load_industry_map()
    industries = sorted(set(industry_map.values()))
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    processed['industry_id'] = processed['股票代码'].map(
        lambda c: industry_to_idx.get(industry_map.get(int(c), ''), 0)
    ).astype('int32')

    # Filter only target dates
    target_dt = pd.to_datetime(target_dates_str)
    target_df = processed[processed['日期'].isin(target_dt)].copy()
    print(f"  target rows: {len(target_df)} (expected {len(target_dates_str) * 300})")

    # Build features per stock (60d window, label skipped)
    feature_names = joblib.load(feature_names_path)
    print(f"  features: {len(feature_names)} (from {feature_names_path.name})")
    cols_to_use = TIER1_LAST_ONLY + TIER2_ENRICHED + TIER3_INDUSTRY
    cols_present = [c for c in cols_to_use if c in processed.columns]

    # Build per-stock windows
    target_dates_set = set(target_dates_str)
    target_idx = {d: i for i, d in enumerate(target_dates_str)}
    all_stocks = sorted(df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_stocks)}

    out = np.full((len(target_dates_str), len(all_stocks), len(feature_names)), np.nan, dtype=np.float32)
    out_dates = []
    out_stocks = []

    for sid, g in processed.groupby('股票代码', sort=False):
        g = g.sort_values('日期').reset_index(drop=True)
        g_dates = g['日期'].dt.strftime('%Y-%m-%d').values
        sid_int = int(sid)
        if sid_int not in stockid2idx:
            continue
        sidx = stockid2idx[sid_int]

        # Find target dates in this stock's data
        for d in g_dates:
            if d in target_dates_set:
                d_idx = target_idx[d]
                # 60d window: [d-60, d-1] (60 days before d)
                d_pos = np.where(g_dates == d)[0][0]
                if d_pos < 60:
                    continue  # not enough history
                window = g.iloc[d_pos-60:d_pos][cols_present].values.astype(np.float32)
                window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
                # Last values per feature (matches enrich_window_factors_np logic)
                feats = np.zeros(len(feature_names), dtype=np.float32)
                for j, c in enumerate(feature_names):
                    if c in cols_present:
                        ci = cols_present.index(c)
                        feats[j] = window[-1, ci]
                out[d_idx, sidx] = feats
                out_dates.append(d)
                out_stocks.append(sid_int)

    print(f"  built features for {len(out_dates)} cells")
    return out, feature_names, scaler_path


def main():
    base_dir = INTEGRATED_DIR / 'tree'         # artifacts (scaler, oof_scores)
    model_dir = MODEL_INTEGRATED_DIR / 'tree'  # model weights
    target_dates = ['2026-06-22', '2026-06-23', '2026-06-24', '2026-06-25', '2026-06-26']
    print(f"[Target] 重新 inference Tree 对 {target_dates}")

    # Build features
    out, feature_names, scaler_path = build_features_for_dates(
        target_dates, base_dir / 'scaler.pkl', base_dir / 'feature_names.pkl'
    )

    # Load scaler and transform
    scaler = joblib.load(scaler_path)
    n_dates, n_stocks, n_features = out.shape
    out_flat = out.reshape(-1, n_features)
    # Only transform valid rows
    valid_mask = ~np.isnan(out_flat).any(axis=1)
    print(f"  valid cells: {valid_mask.sum()}/{len(out_flat)}")
    if valid_mask.sum() > 0:
        out_flat[valid_mask] = scaler.transform(out_flat[valid_mask])
    out_scaled = out_flat.reshape(n_dates, n_stocks, n_features)

    # Load 3 regime Tree models, predict
    REGIME_NAMES = ['bear', 'sideways', 'bull']
    for r in [0, 1, 2]:
        r_name = REGIME_NAMES[r]
        model_path = model_dir / f'model_{r_name}.txt'
        if not model_path.exists():
            print(f"  [{r_name}] model not found, skip")
            continue
        booster = lgb.Booster(model_file=str(model_path))
        # Predict per date
        preds = np.full((n_dates, n_stocks), np.nan, dtype=np.float32)
        for d_idx in range(n_dates):
            day_feats = out_scaled[d_idx]
            day_valid = ~np.isnan(day_feats).any(axis=1)
            if day_valid.sum() > 0:
                preds[d_idx, day_valid] = booster.predict(day_feats[day_valid])
        # Update oof_scores_{r}.npy: replace last 5 days (idx 109-113)
        oof_p = base_dir / f'oof_scores_{r_name}.npy'
        oof = np.load(oof_p)
        print(f"  [{r_name}] before: last 5 days valid = "
              f"{sum((~np.isnan(oof[i])).sum() for i in range(-5, 0))}")
        oof[-5:] = preds  # replace
        np.save(oof_p, oof)
        new_valid = sum((~np.isnan(oof[i])).sum() for i in range(-5, 0))
        print(f"  [{r_name}] after:  last 5 days valid = {new_valid}")
        print(f"  [{r_name}] saved to {oof_p}")


if __name__ == '__main__':
    main()
