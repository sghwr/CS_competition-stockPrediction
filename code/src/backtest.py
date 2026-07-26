"""Backtest: 4 metrics ONLY (no final score, no IR, no Rank-IC).

严格 OOS 回测, 评估 test 段 (6 月, 2026-01-05 ~ 2026-06-26).

评估指标 (4):
  1. abs_return:       绝对组合日均收益率 (top-K 等权, T+1~T+horizon open)
  2. excess_return:    超额收益率 = abs_return - HS300
  3. top5_ratio:       绝对收益率 / 前 5 名股票平均收益率 (>= 1 表明我们抓到了赢家)
  4. ndcg5:            NDCG@5 (预测 top-5 排名 vs 实际 top-5 by 5d return)
  - final_oof / tree_oof / linear_oof 来自 test 段 OOF
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PROJECT_ROOT, INTEGRATED_DIR, TRAIN_CSV, INDEX_CSV


def load_data(base_dir=None):
    """加载 OOF (test 段) + stock_data.csv 价格 + HS300 大盘."""
    if base_dir is None:
        base_dir = INTEGRATED_DIR
    else:
        base_dir = Path(base_dir)
        if not base_dir.is_absolute():
            base_dir = PROJECT_ROOT / base_dir
    print(f"[Base] {base_dir}", flush=True)
    print("[Load] OOF scores (test 段, 3 月) ...", flush=True)
    # 加载 test_meta (3 月 test 段, 来自 macro_transformer/test_meta.json)
    test_meta_path = base_dir / 'macro_transformer' / 'test_meta.json'
    if not test_meta_path.exists():
        raise FileNotFoundError(f"Run train_transformer_branch.py first: missing {test_meta_path}")
    with open(test_meta_path) as f:
        test_meta = json.load(f)
    test_dates = test_meta['dates']
    print(f"  test dates: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)} days)", flush=True)

    final_oof = np.load(base_dir / 'ensemble' / 'final_oof_scores.npy')
    # MacroTransformer: 不输出 per-stock 分数, 用 macro_pred 切片 test 段
    macro_npz = base_dir / 'macro_transformer' / 'macro_pred_seed42.npz'
    if not macro_npz.exists():
        macro_npz = base_dir / 'macro_transformer' / 'macro_pred_avg.npz'
    if macro_npz.exists():
        from industry_mapping import load_industry_map
        with np.load(macro_npz, allow_pickle=True) as z:
            t_dates_all = list(z['dates'])
            ib_all = z['industry_bias']  # (T, 11)
            industries = list(z['industries'])
        t_date_to_idx = {d: i for i, d in enumerate(t_dates_all)}
        test_indices = [t_date_to_idx[d] for d in test_dates if d in t_date_to_idx]
        ib = ib_all[test_indices]  # (test_T, 11)
        industry_map = load_industry_map()
        industry_to_idx = {ind: i for i, ind in enumerate(industries)}
        all_stocks_full = sorted(pd.read_csv(TRAIN_CSV, encoding='utf-8-sig', usecols=['股票代码'])['股票代码'].unique())
        stock_ind = np.array([industry_to_idx.get(industry_map.get(int(s), ''), 0) for s in all_stocks_full])
        transformer_oof = ib[:, stock_ind]  # (test_T, 300)
    else:
        transformer_oof = np.load(base_dir / 'transformer' / 'oof_scores.npy')
    tree_oof = np.load(base_dir / 'tree' / 'oof_scores.npy')
    linear_oof = np.load(base_dir / 'linear_regression' / 'oof_scores_seed42.npy')
    print(f"  final: {final_oof.shape}, transformer: {transformer_oof.shape}, "
          f"tree: {tree_oof.shape}, linear: {linear_oof.shape}", flush=True)
    print(f"  OOF: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)} days)", flush=True)

    print("\n[Load] stock_data.csv prices (唯一价格源) ...", flush=True)
    prices = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    prices = prices.rename(columns={'日期': 'date', '股票代码': 'stock'})
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.sort_values(['stock', 'date']).reset_index(drop=True)
    print(f"  prices: {len(prices):,} rows, {prices['stock'].nunique()} stocks, "
          f"{prices['date'].min().date()} ~ {prices['date'].max().date()}", flush=True)

    print("\n[Load] HS300 index ...", flush=True)
    hs300 = pd.read_csv(INDEX_CSV)
    hs300['date'] = pd.to_datetime(hs300['date'])
    hs300 = hs300.sort_values('date').reset_index(drop=True)
    print(f"  HS300: {len(hs300)} days, "
          f"{hs300['date'].min().date()} ~ {hs300['date'].max().date()}", flush=True)

    return final_oof, transformer_oof, tree_oof, linear_oof, test_dates, prices, hs300


def build_price_lookups(prices, hs300):
    """Build per-day price dicts for fast lookup."""
    prices['_d_str'] = prices['date'].dt.strftime('%Y-%m-%d')
    price_open = {}
    for _, row in prices[['stock', '_d_str', '开盘']].iterrows():
        s = int(row['stock'])
        d = row['_d_str']
        o = row['开盘']
        if pd.notna(o):
            price_open[(s, d)] = float(o)
    print(f"  price_open: {len(price_open):,} entries", flush=True)

    hs300['_d_str'] = hs300['date'].dt.strftime('%Y-%m-%d')
    hs300_open = dict(zip(hs300['_d_str'], hs300['open']))
    print(f"  HS300 open: {len(hs300_open):,} days", flush=True)
    return price_open, hs300_open


def get_actual_5d_return(d_signal, d_buy, d_sell, all_stocks, price_open, top_n=5):
    """Compute actual top-N stock 5d return for a given signal day.

    Returns: (top_n_stocks, top_n_return) where top_n_return is mean of their
    T+1 -> T+5 return. Returns (None, None) if data missing.
    """
    rets = []
    valid_stocks = []
    for s in all_stocks:
        p_buy = price_open.get((s, d_buy))
        p_sell = price_open.get((s, d_sell))
        if p_buy and p_sell and p_buy > 1e-4:
            r = (p_sell - p_buy) / p_buy
            rets.append(r)
            valid_stocks.append((s, r))
    if not rets:
        return None, None
    valid_stocks.sort(key=lambda x: -x[1])  # descending
    top_n_stocks = [s for s, _ in valid_stocks[:top_n]]
    top_n_return = float(np.mean([r for _, r in valid_stocks[:top_n]]))
    return top_n_stocks, top_n_return


def ndcg_at_k(predicted_top_k, actual_top_k, k=5):
    """NDCG@k for ranking: predicted top-k vs actual top-k by return.

    predicted_top_k / actual_top_k: list of stock codes.
    Returns: float in [0, 1].
    """
    actual_set = set(actual_top_k[:k])
    dcg = 0.0
    for rank, s in enumerate(predicted_top_k[:k]):
        if s in actual_set:
            dcg += 1.0 / np.log2(rank + 2)
    # IDCG: actual top-k all in actual_set
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(actual_set))))
    if idcg < 1e-9:
        return 0.0
    return dcg / idcg


def backtest_one(oof, val_dates, all_stocks, price_open, hs300_open,
                  top_k=5, horizon=5, name='branch', selection_mode='industry_diversified'):
    """Run backtest and compute 4 metrics.

    selection_mode:
      - 'pure_topk':          原始 top-K by score (容易行业集中)
      - 'industry_diversified': 每行业最多 1 只, 强制行业分散 (改善 win rate)
      - 'risk_parity':        top-K + 按 inverse volatility 加权
    """
    n_days = oof.shape[0]
    print(f"\n[Backtest:{name}] === top_k={top_k}, horizon={horizon}, "
          f"selection={selection_mode}, n_days={n_days} ===", flush=True)

    # Pre-compute industry map and recent volatility (for risk parity)
    from industry_mapping import load_industry_map
    im = load_industry_map()
    industries = sorted(set(im.values()))
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    stock_industry = np.array([
        industry_to_idx.get(im.get(int(s), ''), 0) for s in all_stocks
    ])

    # Pre-compute 20d rolling vol per stock (for risk parity)
    print(f"  computing 20d volatility for risk parity ...", flush=True)
    import pandas as pd
    from paths import TRAIN_CSV
    df_prices = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig',
                             usecols=['股票代码', '日期', '开盘'])
    df_prices['日期'] = pd.to_datetime(df_prices['日期'])
    df_prices = df_prices.sort_values(['股票代码', '日期']).reset_index(drop=True)
    df_prices['log_ret'] = df_prices.groupby('股票代码')['开盘'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    vol_map = {}
    for sid, grp in df_prices.groupby('股票代码'):
        if len(grp) >= 21:
            v = grp['log_ret'].rolling(20, min_periods=10).std().iloc[-1]
            vol_map[int(sid)] = float(v) if pd.notna(v) else 0.02
        else:
            vol_map[int(sid)] = 0.02
    stock_vol = np.array([vol_map.get(int(s), 0.02) for s in all_stocks])

    portfolio_returns = []
    excess_returns = []
    top5_actuals = []
    ndcg5_scores = []
    valid_dates = []

    for d_idx in range(n_days - horizon):
        d_signal = val_dates[d_idx]
        d_buy = val_dates[d_idx + 1]
        d_sell = val_dates[d_idx + horizon]

        # Pick top-K by score (skip NaN)
        day_scores = oof[d_idx].copy()
        valid_mask = ~np.isnan(day_scores)
        if valid_mask.sum() < top_k:
            continue
        # argsort descending
        idx_sorted = np.argsort(np.where(valid_mask, day_scores, -np.inf))[::-1]

        # Selection logic
        if selection_mode == 'industry_diversified':
            # 每行业最多 1 只: greedy 选 top-by-score, 跳过已选行业
            top_k_stocks = []
            used_industries = set()
            for i in idx_sorted:
                s = all_stocks[i]
                if not valid_mask[i]:
                    continue
                ind = stock_industry[i]
                if ind in used_industries:
                    continue
                top_k_stocks.append(s)
                used_industries.add(ind)
                if len(top_k_stocks) >= top_k:
                    break
        elif selection_mode == 'risk_parity':
            # Risk parity: top-K + inverse vol weights
            top_k_stocks = [all_stocks[i] for i in idx_sorted[:top_k] if valid_mask[i]]
        else:  # 'pure_topk'
            top_k_stocks = [all_stocks[i] for i in idx_sorted[:top_k] if valid_mask[i]]

        if len(top_k_stocks) < 1:
            continue

        # Portfolio return: depends on selection mode
        stock_rets_dict = {}
        for s in top_k_stocks:
            p_buy = price_open.get((s, d_buy))
            p_sell = price_open.get((s, d_sell))
            if p_buy and p_sell and p_buy > 1e-4:
                stock_rets_dict[s] = (p_sell - p_buy) / p_buy

        if not stock_rets_dict:
            continue

        if selection_mode == 'risk_parity':
            # 按 inverse volatility 加权
            weights = {}
            total_inv_vol = 0
            for s in stock_rets_dict:
                idx = all_stocks.index(s)
                w = 1.0 / max(stock_vol[idx], 1e-4)
                weights[s] = w
                total_inv_vol += w
            port_ret = sum(stock_rets_dict[s] * weights[s] for s in stock_rets_dict) / total_inv_vol
        else:
            # Equal weight (industry_diversified or pure_topk)
            port_ret = float(np.mean(list(stock_rets_dict.values())))

        # HS300 return
        p_buy_m = hs300_open.get(d_buy)
        p_sell_m = hs300_open.get(d_sell)
        if not (p_buy_m and p_sell_m and p_buy_m > 1e-4):
            continue
        mkt_ret = (p_sell_m - p_buy_m) / p_buy_m

        # Actual top-5 by 5d return
        actual_top5, top5_ret = get_actual_5d_return(
            d_signal, d_buy, d_sell, all_stocks, price_open, top_n=5,
        )
        if actual_top5 is None:
            continue

        # NDCG@5
        ndcg = ndcg_at_k(top_k_stocks, actual_top5, k=5)

        portfolio_returns.append(port_ret)
        excess_returns.append(port_ret - mkt_ret)
        top5_actuals.append(top5_ret)
        ndcg5_scores.append(ndcg)
        valid_dates.append(d_signal)

    arr_p = np.array(portfolio_returns)
    arr_e = np.array(excess_returns)
    arr_t5 = np.array(top5_actuals)
    arr_n = np.array(ndcg5_scores)
    n = len(arr_p)
    if n == 0:
        print(f"  [WARN] no valid windows", flush=True)
        return None, []

    # Metric 1: 绝对组合收益率 (mean per-window)
    abs_return = float(arr_p.mean())
    # Metric 2: 超额收益率
    excess_return = float(arr_e.mean())
    # Metric 3: 绝对 / 前5名比 (>= 1 = we beat the actual top-5)
    # For each window: ratio = port_ret / top5_ret
    safe_top5 = np.where(np.abs(arr_t5) > 1e-9, arr_t5, np.nan)
    ratios = arr_p / safe_top5
    ratios = ratios[~np.isnan(ratios)]
    top5_ratio = float(ratios.mean()) if len(ratios) > 0 else 0.0
    # Metric 4: NDCG@5
    ndcg5 = float(arr_n.mean())

    print(f"\n[Backtest:{name}] ===== 4 metrics (n={n}) =====", flush=True)
    print(f"  1. 绝对组合收益率:        {abs_return*100:+.4f}% /day (mean)", flush=True)
    print(f"  2. 超额收益率:            {excess_return*100:+.4f}% /day (vs HS300)", flush=True)
    print(f"  3. 绝对收益/前5名比:      {top5_ratio:+.4f} (>= 1 = 抓到了赢家)", flush=True)
    print(f"  4. NDCG@5:                {ndcg5:.4f} (排序质量)", flush=True)
    print(f"  --- 辅助统计 (不计入主指标) ---", flush=True)
    print(f"  累计绝对: {(1 + arr_p).prod() - 1:+.3%}", flush=True)
    print(f"  累计超额: {(1 + arr_e).prod() - 1:+.3%}", flush=True)
    print(f"  累计前5:  {(1 + arr_t5).prod() - 1:+.3%}", flush=True)
    print(f"  超额>0 窗口: {(arr_e > 0).sum()}/{n} = {(arr_e > 0).mean()*100:.1f}%", flush=True)

    summary = {
        'abs_return': abs_return,
        'excess_return': excess_return,
        'top5_ratio': top5_ratio,
        'ndcg5': ndcg5,
        'n_windows': n,
        # 辅助
        'cum_abs': float((1 + arr_p).prod() - 1),
        'cum_excess': float((1 + arr_e).prod() - 1),
        'cum_top5': float((1 + arr_t5).prod() - 1),
        'excess_win_rate': float((arr_e > 0).mean()),
    }
    details = [{'date': d, 'port': float(p), 'excess': float(e),
                'top5': float(t), 'ndcg5': float(n_)}
               for d, p, e, t, n_ in zip(valid_dates, arr_p, arr_e, arr_t5, arr_n)]
    return summary, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5, help='每日持仓股票数')
    parser.add_argument('--horizon', type=int, default=5, help='持有天数')
    parser.add_argument('--selection', type=str, default='industry_diversified',
                        choices=['pure_topk', 'industry_diversified', 'risk_parity'],
                        help='选股策略: pure_topk (原版) / industry_diversified (行业分散) / risk_parity (波动率倒数加权)')
    parser.add_argument('--output', type=str, default='output/integrated_v1/backtest')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='OOF 所在根目录 (默认 INTEGRATED_DIR)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    final_oof, t_oof, g_oof, l_oof, val_dates, prices, hs300 = load_data(args.base_dir)
    all_stocks = sorted(prices['stock'].unique())
    print(f"  all_stocks: {len(all_stocks)}", flush=True)
    price_open, hs300_open = build_price_lookups(prices, hs300)

    results = {}
    # linear 是 stack 的 component, 不独立选股. 只评估 final/tree/transformer.
    # linear 的 IC / importance 在 stack 集成里已经体现.
    for name, oof in [('final', final_oof), ('transformer', t_oof),
                      ('tree', g_oof)]:
        print(f"\n{'=' * 80}\n[Branch] {name}\n{'=' * 80}", flush=True)
        summary, details = backtest_one(
            oof, val_dates, all_stocks, price_open, hs300_open,
            top_k=args.top_k, horizon=args.horizon, name=name,
            selection_mode=args.selection,
        )
        if summary is not None:
            results[name] = summary
        if name == 'final':
            final_details = details

    # linear component 报告 (不作为独立选股策略)
    print(f"\n{'=' * 80}\n[Stack Component] linear_reversal (NOT for individual selection)\n{'=' * 80}", flush=True)
    print("  说明: linear 是 stack model 的 component, 不独立选股.", flush=True)
    print("        在 stack 里通过 linear_pred + linear_x_macro interaction 起作用.", flush=True)
    summary_lin, _ = backtest_one(
        l_oof, val_dates, all_stocks, price_open, hs300_open,
        top_k=args.top_k, horizon=args.horizon, name='linear_reversal (component)',
        selection_mode=args.selection,
    )
    if summary_lin is not None:
        results['linear_reversal_component'] = summary_lin

    # Save
    with open(output_dir / 'backtest_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'config': vars(args), 'results': results,
                    'metrics_used': ['abs_return', 'excess_return', 'top5_ratio', 'ndcg5']},
                   f, indent=2, ensure_ascii=False)
    if final_details:
        pd.DataFrame(final_details).to_csv(
            output_dir / 'final_daily_details.csv', index=False,
            encoding='utf-8-sig')
    print(f"\n[Save] {output_dir}/backtest_summary.json + final_daily_details.csv", flush=True)


if __name__ == '__main__':
    main()
