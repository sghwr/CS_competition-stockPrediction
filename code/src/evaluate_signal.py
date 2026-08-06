"""Evaluate base and manual portfolios for one frozen signal date."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval_metrics import (
    evaluate_portfolio,
    load_benchmark_open_prices,
    load_stock_open_prices,
    normalize_stock_id,
    stock_forward_returns,
)
from manual_policy import adjust_scores
from paths import HS300_LIST_CSV, INDEX_CSV, PROJECT_ROOT, STOCK_DATA_CSV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--buy-date", required=True)
    parser.add_argument("--sell-date", required=True)
    parser.add_argument(
        "--policy",
        default=str(PROJECT_ROOT / "code" / "config" / "manual_policy.json"),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    output_path = Path(args.output) if args.output else prediction_dir / "evaluation.json"
    stock_open = load_stock_open_prices(STOCK_DATA_CSV)
    index_open = load_benchmark_open_prices(INDEX_CSV)
    forward_returns = stock_forward_returns(stock_open, args.buy_date, args.sell_date)

    names = pd.read_csv(
        HS300_LIST_CSV,
        dtype={"code": str},
        encoding="utf-8-sig",
    )
    names["stock_id"] = names["code"].map(normalize_stock_id)
    names = names[["stock_id", "code_name"]].drop_duplicates("stock_id")

    scores = pd.read_csv(prediction_dir / "base_scores.csv", dtype={"stock_id": str})
    adjusted = adjust_scores(
        scores,
        args.signal_date,
        STOCK_DATA_CSV,
        policy_path=args.policy,
    )
    score_columns = [
        "stock_id",
        "score",
        "adjusted_score",
        "industry",
        "past5",
        "vol20",
    ]

    result: dict[str, object] = {}
    for variant in ("base", "manual"):
        portfolio = pd.read_csv(
            prediction_dir / f"portfolio_{variant}.csv",
            dtype={"stock_id": str},
        )
        metric = evaluate_portfolio(
            portfolio,
            args.signal_date,
            args.buy_date,
            args.sell_date,
            stock_open,
            index_open,
            compute_ndcg=True,
        )
        details = (
            portfolio
            .merge(forward_returns, on="stock_id", how="left")
            .merge(names, on="stock_id", how="left")
            .merge(adjusted[score_columns], on="stock_id", how="left")
        )
        details["contribution"] = details["weight"] * details["return"]
        result[variant] = {
            "metrics": metric.__dict__,
            "holdings": details.to_dict("records"),
        }

    actual_top5 = (
        forward_returns
        .sort_values("return", ascending=False)
        .head(5)
        .merge(names, on="stock_id", how="left")
    )
    result["actual_top5"] = actual_top5.to_dict("records")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[Save] {output_path}")


if __name__ == "__main__":
    main()
