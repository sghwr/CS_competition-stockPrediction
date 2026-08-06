"""Trading-date helpers for walk-forward experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EvaluationWindow:
    signal_date: str
    train_label_end: str
    buy_date: str
    sell_date: str


def load_trading_dates(stock_csv: str | Path) -> list[str]:
    df = pd.read_csv(stock_csv, usecols=["日期"], encoding="utf-8-sig")
    dates = pd.to_datetime(df["日期"], errors="coerce").dropna()
    return [d.strftime("%Y-%m-%d") for d in sorted(dates.unique())]


def shift_trade_date(trading_dates: list[str], date: str, offset: int) -> str:
    if date not in trading_dates:
        raise ValueError(f"{date} is not in trading calendar")
    idx = trading_dates.index(date) + offset
    if idx < 0 or idx >= len(trading_dates):
        raise ValueError(f"Cannot shift {date} by {offset}: out of range")
    return trading_dates[idx]


def previous_or_same_trade_date(trading_dates: list[str], date: str) -> str:
    target = pd.Timestamp(date)
    candidates = [d for d in trading_dates if pd.Timestamp(d) <= target]
    if not candidates:
        raise ValueError(f"No trading date <= {date}")
    return candidates[-1]


def make_evaluation_window(trading_dates: list[str], signal_date: str, horizon: int = 5) -> EvaluationWindow:
    signal = previous_or_same_trade_date(trading_dates, signal_date)
    return EvaluationWindow(
        signal_date=signal,
        train_label_end=shift_trade_date(trading_dates, signal, -horizon),
        buy_date=shift_trade_date(trading_dates, signal, 1),
        sell_date=shift_trade_date(trading_dates, signal, horizon),
    )


def iter_non_overlapping_signal_dates(
    trading_dates: list[str],
    start_date: str,
    end_date: str,
    step: int = 5,
    horizon: int = 5,
) -> list[str]:
    start = previous_or_same_trade_date(trading_dates, start_date)
    end = previous_or_same_trade_date(trading_dates, end_date)
    start_idx = trading_dates.index(start)
    end_idx = trading_dates.index(end)
    last_signal_idx = min(end_idx, len(trading_dates) - horizon - 1)
    return trading_dates[start_idx:last_signal_idx + 1:step]
