"""Validate competition result.csv format."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval_metrics import normalize_stock_id
from paths import HS300_LIST_CSV


def validate_result(path: str | Path, universe_path: str | Path = HS300_LIST_CSV) -> list[str]:
    errors: list[str] = []
    result = pd.read_csv(path, dtype={"stock_id": str})
    if list(result.columns) != ["stock_id", "weight"]:
        errors.append("Header must be exactly: stock_id,weight")
    if len(result) > 5:
        errors.append("Result must contain at most 5 rows")
    if "stock_id" in result.columns:
        result["stock_id"] = result["stock_id"].map(normalize_stock_id)
        if result["stock_id"].duplicated().any():
            errors.append("Duplicate stock_id found")
        bad_code = ~result["stock_id"].str.match(r"^\d{6}$", na=False)
        if bad_code.any():
            errors.append("All stock_id values must be 6-digit codes")

    if "weight" in result.columns:
        weight = pd.to_numeric(result["weight"], errors="coerce")
        if weight.isna().any():
            errors.append("All weights must be numeric")
        if (weight < -1e-12).any():
            errors.append("Weights must be non-negative")
        if weight.sum() > 1.0 + 1e-8:
            errors.append(f"Weight sum must be <= 1.0, got {weight.sum():.12f}")

    universe = pd.read_csv(universe_path, dtype={"code": str}, encoding="utf-8-sig")
    allowed = set(universe["code"].map(normalize_stock_id))
    unknown = sorted(set(result.get("stock_id", pd.Series(dtype=str))) - allowed)
    if unknown:
        errors.append(f"Unknown stock_id outside HS300 list: {unknown[:10]}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--universe", default=str(HS300_LIST_CSV))
    args = parser.parse_args()
    errors = validate_result(args.result, args.universe)
    if errors:
        print("[FAIL]")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("[OK] result.csv format is valid")


if __name__ == "__main__":
    main()
