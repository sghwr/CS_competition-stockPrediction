"""Predict and build a portfolio for one as-of signal date."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from integrated_model import (
    REGIME_NAMES,
    STACK_FEATURES_PER_REGIME,
    build_explicit_signals,
    get_regime_for_dates,
    zscore_per_day,
)
from industry_mapping import load_industry_map
from manual_policy import adjust_scores, build_portfolio_from_scores, load_policy
from mdsrp_regression import build_reversal_features
from model import LinearRegressionModel
from paths import PROJECT_ROOT, STOCK_DATA_CSV
from tree_branch import add_industry_features, enrich_window_factors_np
from utils import TIER1_LAST_ONLY, TIER2_ENRICHED, TIER3_INDUSTRY, engineer_features_158plus39


def _normalize_stock_id(value: object) -> str:
    return str(value).replace("sh.", "").replace("sz.", "").split(".")[0].zfill(6)


def _load_stock_universe() -> list[int]:
    df = pd.read_csv(STOCK_DATA_CSV, encoding="utf-8-sig", usecols=["股票代码"])
    return sorted(df["股票代码"].astype(int).unique().tolist())


def _zscore_one_day(values: np.ndarray) -> np.ndarray:
    return zscore_per_day(values.reshape(1, -1))[0]


def _load_torch_state(path: Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_tree_features(signal_date: str, branch_dir: Path, all_stocks: list[int]) -> np.ndarray:
    feature_names = joblib.load(branch_dir / "feature_names.pkl")
    df = pd.read_csv(STOCK_DATA_CSV, encoding="utf-8-sig")
    df["日期"] = pd.to_datetime(df["日期"])
    signal_ts = pd.Timestamp(signal_date)
    df = df[df["日期"] <= signal_ts].sort_values(["股票代码", "日期"]).reset_index(drop=True)

    groups = [g for _, g in df.groupby("股票代码", sort=False)]
    processed = pd.concat([engineer_features_158plus39(g) for g in groups]).reset_index(drop=True)
    processed = add_industry_features(processed, load_industry_map())
    for col in TIER1_LAST_ONLY + TIER2_ENRICHED + TIER3_INDUSTRY:
        if col in processed.columns:
            processed[col] = processed[col].replace([np.inf, -np.inf], 0).fillna(0)

    cols_to_use = [c for c in TIER1_LAST_ONLY + TIER2_ENRICHED + TIER3_INDUSTRY if c in processed.columns]
    stock_to_idx = {sid: i for i, sid in enumerate(all_stocks)}
    features = np.full((len(all_stocks), len(feature_names)), np.nan, dtype=np.float32)
    for sid, group in processed.groupby("股票代码", sort=False):
        sid_int = int(sid)
        if sid_int not in stock_to_idx:
            continue
        group = group.sort_values("日期").reset_index(drop=True)
        date_values = group["日期"].dt.strftime("%Y-%m-%d").values
        matches = np.where(date_values == signal_date)[0]
        if len(matches) == 0:
            continue
        pos = int(matches[-1])
        if pos < 59:
            continue
        window = group.iloc[pos - 59:pos + 1][cols_to_use].values.astype(np.float32)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        enriched = enrich_window_factors_np(window, cols_to_use)
        features[stock_to_idx[sid_int]] = np.array(
            [enriched.get(name, 0.0) for name in feature_names],
            dtype=np.float32,
        )

    return features


def predict_tree(signal_date: str, base_dir: Path, model_dir: Path, all_stocks: list[int]) -> dict[int, np.ndarray]:
    features = build_tree_features(signal_date, base_dir / "tree", all_stocks)
    valid = ~np.isnan(features).any(axis=1)
    preds: dict[int, np.ndarray] = {}
    for regime, name in enumerate(REGIME_NAMES):
        path = model_dir / "tree" / f"model_{name}.txt"
        if not path.exists():
            continue
        booster = lgb.Booster(model_file=str(path))
        out = np.full(len(all_stocks), np.nan, dtype=np.float32)
        if valid.any():
            out[valid] = booster.predict(features[valid])
        preds[regime] = _zscore_one_day(out)
    if not preds:
        raise FileNotFoundError(f"No tree models under {model_dir / 'tree'}")
    return preds


def predict_linear(signal_date: str, base_dir: Path, model_dir: Path, all_stocks: list[int]) -> np.ndarray:
    branch_dir = base_dir / "linear_regression"
    scaler = joblib.load(branch_dir / "scaler.pkl")
    feature_cols = joblib.load(branch_dir / "feature_names.pkl")
    df = pd.read_csv(STOCK_DATA_CSV, encoding="utf-8-sig")
    df["日期"] = pd.to_datetime(df["日期"])
    signal_ts = pd.Timestamp(signal_date)
    df = df[df["日期"] <= signal_ts].sort_values(["股票代码", "日期"]).reset_index(drop=True)
    processed = build_reversal_features(df)
    target = processed[processed["日期"] == signal_ts].copy()
    if target.empty:
        raise ValueError(f"No stock rows for signal date {signal_date}")

    X = scaler.transform(target[feature_cols].values.astype(np.float32))
    stock_to_idx = {sid: i for i, sid in enumerate(all_stocks)}
    target["_s_idx"] = target["股票代码"].astype(int).map(stock_to_idx)
    valid = target["_s_idx"].notna().values
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_preds = []
    for path in sorted((model_dir / "linear_regression").glob("best_seed*.pth")):
        state = _load_torch_state(path, device)
        n_features = state["linear.weight"].shape[1]
        model = LinearRegressionModel(n_features).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X).float().to(device)).cpu().numpy().reshape(-1)
        mat = np.full(len(all_stocks), np.nan, dtype=np.float32)
        mat[target.loc[valid, "_s_idx"].astype(int).values] = pred[valid]
        seed_preds.append(mat)
    if not seed_preds:
        raise FileNotFoundError(f"No linear checkpoints under {model_dir / 'linear_regression'}")
    return _zscore_one_day(np.nanmean(seed_preds, axis=0))


def load_macro_for_date(signal_date: str, base_dir: Path) -> dict:
    path = base_dir / "macro_transformer" / "macro_pred_avg.npz"
    if not path.exists():
        files = sorted((base_dir / "macro_transformer").glob("macro_pred_seed*.npz"))
        if not files:
            raise FileNotFoundError(f"No macro predictions under {base_dir / 'macro_transformer'}")
        path = files[0]
    npz = np.load(path, allow_pickle=True)
    dates = [str(d) for d in npz["dates"]]
    if signal_date in dates:
        idx = dates.index(signal_date)
    else:
        eligible = [i for i, d in enumerate(dates) if d <= signal_date]
        if not eligible:
            raise ValueError(f"No macro prediction <= {signal_date}")
        idx = eligible[-1]
    return {
        "date_used": dates[idx],
        "industry_bias": npz["industry_bias"][idx],
        "industry_beta": npz["industry_beta"][idx],
        "industries": [str(x) for x in npz["industries"]],
    }


def stack_predict(
    signal_date: str,
    base_dir: Path,
    model_dir: Path,
    all_stocks: list[int],
) -> np.ndarray:
    tree_preds = predict_tree(signal_date, base_dir, model_dir, all_stocks)
    linear_pred = predict_linear(signal_date, base_dir, model_dir, all_stocks)
    macro = load_macro_for_date(signal_date, base_dir)
    signals = build_explicit_signals([signal_date], all_stocks)
    regime = int(get_regime_for_dates([signal_date])[0])

    industry_map = load_industry_map()
    industries = sorted(set(industry_map.values()))
    industry_to_idx = {industry: i for i, industry in enumerate(industries)}
    stock_industry = np.array([
        industry_to_idx.get(industry_map.get(int(stock), ""), 0)
        for stock in all_stocks
    ], dtype=np.int32)
    macro_ind_to_idx = {industry: i for i, industry in enumerate(macro["industries"])}
    macro_bias_by_industry = np.zeros(len(industries), dtype=np.float32)
    for industry, idx in industry_to_idx.items():
        macro_idx = macro_ind_to_idx.get(industry)
        if macro_idx is not None:
            macro_bias_by_industry[idx] = float(macro["industry_bias"][macro_idx])

    models = {}
    for r, name in enumerate(REGIME_NAMES):
        path = model_dir / "ensemble" / f"stack_{name}.pkl"
        if path.exists():
            models[r] = joblib.load(path)
    model = models.get(regime) or models.get(1) or models.get(2) or models.get(0)
    if model is None:
        raise FileNotFoundError(f"No stack models under {model_dir / 'ensemble'}")

    tree_pred = tree_preds.get(regime)
    if tree_pred is None:
        tree_pred = tree_preds.get(1)
    if tree_pred is None:
        tree_pred = next(iter(tree_preds.values()))
    feature_names = STACK_FEATURES_PER_REGIME[regime]
    X = np.zeros((len(all_stocks), len(feature_names)), dtype=np.float32)
    for s_idx, sid in enumerate(stock_industry):
        col = 0
        X[s_idx, col] = 0.0 if np.isnan(tree_pred[s_idx]) else tree_pred[s_idx]; col += 1
        X[s_idx, col] = 0.0 if np.isnan(linear_pred[s_idx]) else linear_pred[s_idx]; col += 1
        X[s_idx, col] = macro_bias_by_industry[sid]; col += 1
        X[s_idx, col] = sid; col += 1
        for signal_name in feature_names[col:]:
            X[s_idx, col] = signals.get(signal_name, np.zeros((1, len(all_stocks)), dtype=np.float32))[0, s_idx]
            col += 1
    return model.predict(X)


def _base_policy(top_k: int, selection: str) -> dict:
    policy = load_policy(None)
    policy["enabled"] = False
    policy["top_k"] = 1 if selection == "single" else top_k
    policy["gross_exposure"] = 1.0
    policy["max_single_weight"] = 1.0 if selection == "single" else 1.0 / max(1, top_k)
    policy["max_per_industry"] = 1 if selection == "industry_diversified" else top_k
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--policy", default=str(PROJECT_ROOT / "code" / "config" / "manual_policy.json"))
    parser.add_argument("--selection", choices=["pure_topk", "industry_diversified", "single"], default="industry_diversified")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stocks = _load_stock_universe()
    scores = stack_predict(args.signal_date, base_dir, model_dir, all_stocks)
    score_df = pd.DataFrame({
        "stock_id": [_normalize_stock_id(s) for s in all_stocks],
        "score": scores.astype(float),
    }).sort_values("score", ascending=False)
    score_df.to_csv(output_dir / "base_scores.csv", index=False, encoding="utf-8")

    base_adjusted = adjust_scores(score_df, args.signal_date, STOCK_DATA_CSV, policy_path=None)
    base_portfolio = build_portfolio_from_scores(base_adjusted, _base_policy(args.top_k, args.selection))
    base_portfolio.to_csv(output_dir / "portfolio_base.csv", index=False, encoding="utf-8")

    manual_policy = load_policy(args.policy)
    manual_adjusted = adjust_scores(score_df, args.signal_date, STOCK_DATA_CSV, policy_path=args.policy)
    manual_portfolio = build_portfolio_from_scores(manual_adjusted, manual_policy)
    manual_portfolio.to_csv(output_dir / "portfolio_manual.csv", index=False, encoding="utf-8")

    result = manual_portfolio if manual_policy.get("enabled", False) else base_portfolio
    result.to_csv(output_dir / "result.csv", index=False, encoding="utf-8")
    with open(output_dir / "prediction_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "signal_date": args.signal_date,
            "base_dir": str(base_dir),
            "model_dir": str(model_dir),
            "selection": args.selection,
            "manual_enabled": bool(manual_policy.get("enabled", False)),
        }, f, ensure_ascii=False, indent=2)

    print(f"[Save] {output_dir / 'base_scores.csv'}", flush=True)
    print(f"[Save] {output_dir / 'portfolio_base.csv'}", flush=True)
    print(f"[Save] {output_dir / 'portfolio_manual.csv'}", flush=True)
    print(f"[Save] {output_dir / 'result.csv'}", flush=True)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
