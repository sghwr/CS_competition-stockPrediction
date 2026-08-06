"""Portfolio metrics for the stock-prediction competition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioMetric:
    signal_date: str
    buy_date: str
    sell_date: str
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    excess_win: bool
    ndcg5: float | None


def normalize_stock_id(value: object) -> str:
    return str(value).replace("sh.", "").replace("sz.", "").split(".")[0].zfill(6)


def load_stock_open_prices(stock_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(stock_csv, dtype={"股票代码": str}, encoding="utf-8-sig")
    df["股票代码"] = df["股票代码"].map(normalize_stock_id)
    df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    return df[["股票代码", "日期", "开盘"]].rename(columns={"股票代码": "stock_id", "日期": "date", "开盘": "open"})


def load_benchmark_open_prices(index_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(index_csv, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "open"]]


def stock_forward_returns(open_prices: pd.DataFrame, buy_date: str, sell_date: str) -> pd.DataFrame:
    buy = open_prices[open_prices["date"] == buy_date][["stock_id", "open"]].rename(columns={"open": "buy_open"})
    sell = open_prices[open_prices["date"] == sell_date][["stock_id", "open"]].rename(columns={"open": "sell_open"})
    out = buy.merge(sell, on="stock_id", how="inner")
    out = out[out["buy_open"] > 0].copy()
    out["return"] = (out["sell_open"] - out["buy_open"]) / out["buy_open"]
    return out[["stock_id", "return"]]


def benchmark_return(index_open: pd.DataFrame, buy_date: str, sell_date: str) -> float:
    buy = index_open.loc[index_open["date"] == buy_date, "open"]
    sell = index_open.loc[index_open["date"] == sell_date, "open"]
    if buy.empty or sell.empty or float(buy.iloc[0]) <= 0:
        raise ValueError(f"Missing benchmark open for {buy_date}->{sell_date}")
    return (float(sell.iloc[0]) - float(buy.iloc[0])) / float(buy.iloc[0])


def ndcg_at_5(predicted: list[str], actual_returns: pd.DataFrame) -> float:
    ranked = actual_returns.sort_values("return", ascending=False).reset_index(drop=True)
    top5 = ranked.head(5)["stock_id"].tolist()
    if not top5:
        return 0.0
    dcg = 0.0
    for rank, stock_id in enumerate(predicted[:5]):
        if stock_id in top5:
            dcg += 1.0 / np.log2(rank + 2)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(5, len(top5))))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_portfolio(
    portfolio: pd.DataFrame,
    signal_date: str,
    buy_date: str,
    sell_date: str,
    open_prices: pd.DataFrame,
    index_open: pd.DataFrame,
    compute_ndcg: bool = True,
) -> PortfolioMetric:
    required = {"stock_id", "weight"}
    missing = required - set(portfolio.columns)
    if missing:
        raise ValueError(f"Portfolio missing columns: {sorted(missing)}")
    pf = portfolio.copy()
    pf["stock_id"] = pf["stock_id"].map(normalize_stock_id)
    pf["weight"] = pd.to_numeric(pf["weight"], errors="coerce").fillna(0.0)
    returns = stock_forward_returns(open_prices, buy_date, sell_date)
    merged = pf.merge(returns, on="stock_id", how="left")
    port_ret = float((merged["weight"] * merged["return"].fillna(0.0)).sum())
    bench_ret = benchmark_return(index_open, buy_date, sell_date)
    ndcg = ndcg_at_5(pf["stock_id"].tolist(), returns) if compute_ndcg else None
    excess = port_ret - bench_ret
    return PortfolioMetric(
        signal_date=signal_date,
        buy_date=buy_date,
        sell_date=sell_date,
        portfolio_return=port_ret,
        benchmark_return=bench_ret,
        excess_return=excess,
        excess_win=excess > 0,
        ndcg5=ndcg,
    )


def summarize_metrics(rows: list[PortfolioMetric]) -> dict:
    if not rows:
        return {"n": 0}
    excess = np.array([r.excess_return for r in rows], dtype=float)
    absolute = np.array([r.portfolio_return for r in rows], dtype=float)
    ndcg = np.array([r.ndcg5 for r in rows if r.ndcg5 is not None], dtype=float)
    return {
        "n": int(len(rows)),
        "mean_return": float(absolute.mean()),
        "mean_excess_return": float(excess.mean()),
        "excess_win_rate": float((excess > 0).mean()),
        "median_excess_return": float(np.median(excess)),
        "worst_excess_return": float(excess.min()),
        "cum_excess_return": float(np.prod(1.0 + excess) - 1.0),
        "mean_ndcg5": float(ndcg.mean()) if len(ndcg) else None,
    }
