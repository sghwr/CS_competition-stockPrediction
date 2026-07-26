"""Wrapper: 用现成 Linear model 重新 inference 6.22-6.26 (跳过 label drop)"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from mdsrp_regression import build_reversal_features, REVERSAL_FEATURES, set_seed
from model import LinearRegressionModel
from paths import INTEGRATED_DIR, MODEL_INTEGRATED_DIR, TRAIN_CSV


def main():
    base_dir = INTEGRATED_DIR / 'linear_regression'        # artifacts (scaler, oof_scores)
    model_dir = MODEL_INTEGRATED_DIR / 'linear_regression'  # model weights
    target_dates = ['2026-06-22', '2026-06-23', '2026-06-24', '2026-06-25', '2026-06-26']
    print(f"[Target] 重新 inference Linear 对 {target_dates}")

    # 1. Read data with extended context
    print("\n[Data] Reading stock data...")
    df = pd.read_csv(TRAIN_CSV)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    context_start = pd.Timestamp('2025-12-01')
    target_end = pd.Timestamp(target_dates[-1])
    context = df[(df['日期'] >= context_start) & (df['日期'] <= target_end)].copy()
    print(f"  context: {context['日期'].min().date()} to {context['日期'].max().date()}, "
          f"{len(context)} rows")

    # 2. Build features (no label drop)
    print("\n[Features] building reversal features (no label drop)...")
    processed = build_reversal_features(context)
    print(f"  processed rows: {len(processed)}")

    # 3. Filter target dates
    target_dt = pd.to_datetime(target_dates)
    target_df = processed[processed['日期'].isin(target_dt)].copy()
    print(f"  target rows: {len(target_df)} (expected {len(target_dates) * 300})")

    # 4. Load scaler and feature names
    scaler = joblib.load(base_dir / 'scaler.pkl')
    feature_cols = joblib.load(base_dir / 'feature_names.pkl')
    print(f"  features: {len(feature_cols)}: {feature_cols}")

    # 5. Transform features
    X = target_df[feature_cols].values.astype(np.float32)
    X = scaler.transform(X)
    print(f"  X: {X.shape}, NaN={np.isnan(X).sum()}")

    # 6. Load 3 seed models, predict
    all_stocks_full = sorted(df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_stocks_full)}
    test_dates_arr = target_dates
    test_date_to_idx = {d: i for i, d in enumerate(test_dates_arr)}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_features = X.shape[1]

    for seed in [42, 123, 7]:
        ckpt_p = model_dir / f'best_seed{seed}.pth'
        if not ckpt_p.exists():
            print(f"  [seed{seed}] model not found, skip")
            continue
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        # 直接用 nn.Linear (model 是 single linear layer)
        n_in = ckpt['linear.weight'].shape[1]
        model = LinearRegressionModel(n_in).to(device)
        model.load_state_dict(ckpt)
        model.eval()

        with torch.no_grad():
            X_t = torch.from_numpy(X).to(device)
            preds_t = model(X_t)
            preds = preds_t.cpu().numpy().flatten()

        # Reshape to (5 dates, 300 stocks)
        target_df['_d_str'] = target_df['日期'].dt.strftime('%Y-%m-%d')
        target_df['_d_idx'] = target_df['_d_str'].map(test_date_to_idx)
        target_df['_s_idx'] = target_df['股票代码'].map(stockid2idx)
        valid = target_df['_d_idx'].notna() & target_df['_s_idx'].notna()

        test_oof = np.full((len(test_dates_arr), len(all_stocks_full)), np.nan, dtype=np.float32)
        test_oof[target_df.loc[valid, '_d_idx'].astype(int).values,
                 target_df.loc[valid, '_s_idx'].astype(int).values] = preds[valid.values]

        # Update oof_scores_seed{seed}.npy
        oof_p = base_dir / f'oof_scores_seed{seed}.npy'
        oof = np.load(oof_p)
        print(f"  [seed{seed}] before: last 5 days valid = "
              f"{sum((~np.isnan(oof[i])).sum() for i in range(-5, 0))}")
        oof[-5:] = test_oof
        np.save(oof_p, oof)
        new_valid = sum((~np.isnan(oof[i])).sum() for i in range(-5, 0))
        print(f"  [seed{seed}] after:  last 5 days valid = {new_valid}")
        print(f"  [seed{seed}] saved to {oof_p}")


if __name__ == '__main__':
    main()
