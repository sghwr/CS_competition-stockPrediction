#!/usr/bin/env python3
"""Validate stock/index coverage without reading target-period returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STOCK_ID_ALIASES = ["stock_id", "code", "\u80a1\u7968\u4ee3\u7801"]
DATE_ALIASES = ["date", "\u65e5\u671f"]


def find_column(columns: list[str], aliases: list[str], label: str) -> str:
    for alias in aliases:
        if alias in columns:
            return alias
    raise ValueError(f"Cannot find {label} column; available columns: {columns}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--required-through", required=True)
    parser.add_argument("--expected-stock-count", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    stock_path = root / "data" / "stock_data.csv"
    index_path = root / "data" / "index_data.csv"
    if not stock_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("stock_data.csv and index_data.csv are both required")

    stock = pd.read_csv(stock_path, encoding="utf-8-sig", low_memory=False)
    index = pd.read_csv(index_path, encoding="utf-8-sig", low_memory=False)
    stock_id_col = find_column(list(stock.columns), STOCK_ID_ALIASES, "stock id")
    stock_date_col = find_column(list(stock.columns), DATE_ALIASES, "stock date")
    index_date_col = find_column(list(index.columns), DATE_ALIASES, "index date")

    stock[stock_id_col] = stock[stock_id_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    stock[stock_date_col] = pd.to_datetime(stock[stock_date_col], errors="coerce")
    index[index_date_col] = pd.to_datetime(index[index_date_col], errors="coerce")
    required = pd.Timestamp(args.required_through)

    errors: list[str] = []
    warnings: list[str] = []
    if stock[stock_date_col].isna().any():
        errors.append("stock_data contains invalid dates")
    if index[index_date_col].isna().any():
        errors.append("index_data contains invalid dates")
    duplicate_stock_keys = int(stock.duplicated([stock_id_col, stock_date_col]).sum())
    duplicate_index_dates = int(index.duplicated([index_date_col]).sum())
    if duplicate_stock_keys:
        errors.append(f"stock_data has {duplicate_stock_keys} duplicate stock/date keys")
    if duplicate_index_dates:
        errors.append(f"index_data has {duplicate_index_dates} duplicate dates")

    stock_count = int(stock[stock_id_col].nunique())
    if stock_count != args.expected_stock_count:
        errors.append(f"expected {args.expected_stock_count} stocks, found {stock_count}")
    stock_latest = stock[stock_date_col].max()
    index_latest = index[index_date_col].max()
    if stock_latest < required:
        errors.append(f"stock_data latest date {stock_latest.date()} is before {required.date()}")
    if index_latest < required:
        errors.append(f"index_data latest date {index_latest.date()} is before {required.date()}")

    stock_dates = set(stock[stock_date_col].dropna().dt.normalize())
    index_dates = set(index[index_date_col].dropna().dt.normalize())
    relevant_index_dates = {d for d in index_dates if d <= required}
    missing_global_dates = sorted(relevant_index_dates - stock_dates)
    if missing_global_dates:
        errors.append(f"stock_data misses {len(missing_global_dates)} benchmark trading dates")

    last_by_stock = stock.groupby(stock_id_col)[stock_date_col].max()
    stale = int(((stock_latest - last_by_stock).dt.days > 7).sum())
    if stale:
        warnings.append(f"{stale} stocks have no row within 7 calendar days of stock latest date")

    report = {
        "status": "ok" if not errors else "failed",
        "required_through": required.strftime("%Y-%m-%d"),
        "stock_rows": int(len(stock)),
        "stock_count": stock_count,
        "stock_latest": stock_latest.strftime("%Y-%m-%d"),
        "index_rows": int(len(index)),
        "index_latest": index_latest.strftime("%Y-%m-%d"),
        "duplicate_stock_keys": duplicate_stock_keys,
        "duplicate_index_dates": duplicate_index_dates,
        "errors": errors,
        "warnings": warnings,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
