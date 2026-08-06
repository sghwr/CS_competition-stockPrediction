"""Update HS300 index data for benchmark excess-return evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import baostock as bs
import pandas as pd


DEFAULT_START_DATE = "2019-01-02"
DEFAULT_END_DATE = "2026-07-24"
INDEX_CODE = "sh.000300"


def login() -> None:
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"baostock login failed: {result.error_msg}")


def logout() -> None:
    bs.logout()


def fetch_index_data(start_date: str, end_date: str) -> pd.DataFrame:
    fields = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
    rs = bs.query_history_k_data_plus(
        INDEX_CODE,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"query {INDEX_CODE} failed: {rs.error_msg}")

    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        raise RuntimeError(f"No index rows returned for {start_date}~{end_date}")

    df = pd.DataFrame(rows, columns=rs.fields)
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.drop(columns=["code"])


def recompute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_dt"] = pd.to_datetime(out["date"])
    out = out.sort_values("date_dt").drop_duplicates("date", keep="last").reset_index(drop=True)
    out["return_1d"] = out["close"] / out["close"].shift(1) - 1
    out.loc[out["close"].shift(1).isna(), "return_1d"] = pd.NA
    out["ma5"] = out["close"].rolling(5, min_periods=5).mean()
    out["ma20"] = out["close"].rolling(20, min_periods=20).mean()
    out["volatility_20d"] = out["return_1d"].rolling(20, min_periods=20).std()
    return out[[
        "date",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "pctChg",
        "return_1d",
        "ma5",
        "ma20",
        "volatility_20d",
    ]]


def update_index_csv(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    fetched = fetch_index_data(start_date, end_date)
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig")
        old["date"] = pd.to_datetime(old["date"]).dt.strftime("%Y-%m-%d")
        old = old[(old["date"] < start_date) | (old["date"] > end_date)].copy()
        merged = pd.concat([old, fetched], ignore_index=True)
    else:
        merged = fetched
    merged = recompute_features(merged)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output", default="data/index_data.csv")
    args = parser.parse_args()

    output = Path(args.output)
    login()
    try:
        df = update_index_csv(output, args.start_date, args.end_date)
    finally:
        logout()

    print(f"[OK] saved {len(df)} rows to {output}")
    print(f"[Range] {df['date'].min()} ~ {df['date'].max()}")
    print(df.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
