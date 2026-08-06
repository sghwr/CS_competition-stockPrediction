"""Manual hard-rule overlay for final portfolio construction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_metrics import normalize_stock_id
from paths import DATA_DIR


DEFAULT_POLICY = {
    "enabled": False,
    "top_k": 5,
    "gross_exposure": 1.0,
    "max_single_weight": 0.25,
    "max_per_industry": 2,
    "min_industries": 1,
    "industry_prior": {},
    "preferred_stocks": {},
    "blocked_industries": [],
    "blocked_stocks": [],
    "leader_effect": {"enabled": False, "bonus": 0.0, "top_leader_quantile": 0.8},
    "risk_filter": {
        "avoid_extreme_5d_jump": False,
        "extreme_5d_threshold": 0.18,
        "volatility_penalty": 0.0,
        "overheat_penalty": 0.0,
    },
}


def load_policy(path: str | Path | None) -> dict:
    policy = json.loads(json.dumps(DEFAULT_POLICY))
    if path is None:
        return policy
    p = Path(path)
    if not p.exists():
        return policy
    with open(p, encoding="utf-8") as f:
        user = json.load(f)
    _deep_update(policy, user)
    return policy


def _deep_update(base: dict, user: dict) -> None:
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def load_stock_industry(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else DATA_DIR / "stock_industry.csv"
    df = pd.read_csv(p, dtype={"code": str}, encoding="utf-8-sig")
    df["stock_id"] = df["code"].map(normalize_stock_id)
    return df[["stock_id", "industry"]].drop_duplicates("stock_id")


def build_recent_features(stock_csv: str | Path, asof_date: str) -> pd.DataFrame:
    df = pd.read_csv(stock_csv, dtype={"股票代码": str}, encoding="utf-8-sig")
    df["stock_id"] = df["股票代码"].map(normalize_stock_id)
    df["日期"] = pd.to_datetime(df["日期"])
    asof = pd.Timestamp(asof_date)
    df = df[df["日期"] <= asof].sort_values(["stock_id", "日期"]).copy()
    df["ret_1d"] = df.groupby("stock_id")["收盘"].pct_change()
    df["past5"] = df.groupby("stock_id")["收盘"].pct_change(5)
    df["vol20"] = df.groupby("stock_id")["ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).std())
    last = df.groupby("stock_id", as_index=False).tail(1)
    return last[["stock_id", "past5", "vol20"]].fillna(0.0)


def adjust_scores(
    scores: pd.DataFrame,
    asof_date: str,
    stock_csv: str | Path,
    policy_path: str | Path | None = None,
    industry_path: str | Path | None = None,
) -> pd.DataFrame:
    policy = load_policy(policy_path)
    out = scores.copy()
    out["stock_id"] = out["stock_id"].map(normalize_stock_id)
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(-np.inf)
    out["adjusted_score"] = out["score"].astype(float)
    industries = load_stock_industry(industry_path)
    out = out.merge(industries, on="stock_id", how="left")

    if not policy.get("enabled", False):
        return out

    blocked_stocks = {normalize_stock_id(s) for s in policy.get("blocked_stocks", [])}
    blocked_industries = set(policy.get("blocked_industries", []))
    out.loc[out["stock_id"].isin(blocked_stocks), "adjusted_score"] = -np.inf
    out.loc[out["industry"].isin(blocked_industries), "adjusted_score"] = -np.inf

    industry_prior = policy.get("industry_prior", {})
    out["adjusted_score"] += out["industry"].map(industry_prior).fillna(0.0).astype(float)
    stock_prior = {normalize_stock_id(k): float(v) for k, v in policy.get("preferred_stocks", {}).items()}
    out["adjusted_score"] += out["stock_id"].map(stock_prior).fillna(0.0).astype(float)

    recent = build_recent_features(stock_csv, asof_date)
    out = out.merge(recent, on="stock_id", how="left").fillna({"past5": 0.0, "vol20": 0.0})
    risk = policy.get("risk_filter", {})
    if risk.get("avoid_extreme_5d_jump", False):
        threshold = float(risk.get("extreme_5d_threshold", 0.18))
        out.loc[out["past5"] > threshold, "adjusted_score"] = -np.inf
    out["adjusted_score"] -= float(risk.get("volatility_penalty", 0.0)) * out["vol20"].abs()
    out["adjusted_score"] -= float(risk.get("overheat_penalty", 0.0)) * out["past5"].clip(lower=0.0)

    leader = policy.get("leader_effect", {})
    if leader.get("enabled", False) and float(leader.get("bonus", 0.0)) != 0.0:
        q = float(leader.get("top_leader_quantile", 0.8))
        bonus = float(leader.get("bonus", 0.0))
        thresholds = out.groupby("industry")["past5"].transform(lambda s: s.quantile(q))
        out.loc[out["past5"] >= thresholds, "adjusted_score"] += bonus

    return out


def build_portfolio_from_scores(adjusted: pd.DataFrame, policy: dict | None = None) -> pd.DataFrame:
    policy = policy or DEFAULT_POLICY
    top_k = int(policy.get("top_k", 5))
    gross = max(0.0, min(1.0, float(policy.get("gross_exposure", 1.0))))
    max_single = max(0.0, min(1.0, float(policy.get("max_single_weight", 0.25))))
    max_per_industry = int(policy.get("max_per_industry", top_k))
    picked = []
    industry_counts: dict[str, int] = {}
    ordered = adjusted.sort_values("adjusted_score", ascending=False)
    for _, row in ordered.iterrows():
        if len(picked) >= top_k:
            break
        if not np.isfinite(row["adjusted_score"]):
            continue
        industry = row.get("industry", "")
        if industry_counts.get(industry, 0) >= max_per_industry:
            continue
        picked.append({"stock_id": row["stock_id"], "industry": industry, "score": float(row["adjusted_score"])})
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

    if not picked:
        return pd.DataFrame(columns=["stock_id", "weight"])

    n = len(picked)
    weight = min(max_single, gross / n)
    weights = [weight] * n
    total = sum(weights)
    if total > gross and total > 0:
        weights = [w * gross / total for w in weights]
    return pd.DataFrame({"stock_id": [p["stock_id"] for p in picked], "weight": weights})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="CSV with stock_id,score")
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--stock-csv", default=str(DATA_DIR / "stock_data.csv"))
    parser.add_argument("--policy", default=str(Path(__file__).resolve().parents[1] / "config" / "manual_policy.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, dtype={"stock_id": str})
    policy = load_policy(args.policy)
    adjusted = adjust_scores(scores, args.asof_date, args.stock_csv, args.policy)
    portfolio = build_portfolio_from_scores(adjusted, policy)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    portfolio.to_csv(args.output, index=False, encoding="utf-8")
    print(portfolio.to_string(index=False))


if __name__ == "__main__":
    main()
