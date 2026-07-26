"""
每日收益追踪: 分析 result.csv 选定股票从 6.29 到 7.3 的每日收益
起点 = 6.29 开盘价买入(首日收益 from open to close, 后续 close-to-close)
用法: python code/src/daily_return_tracker.py
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import DATA_DIR, INTEGRATED_DIR

RESULT_PATH = INTEGRATED_DIR / 'result.csv'
STOCK_DATA_PATH = DATA_DIR / 'stock_data.csv'


def load_picks():
    df = pd.read_csv(RESULT_PATH)
    picks, weights = [], []
    for _, row in df.iterrows():
        sid = str(int(float(row['stock_id'])))
        if len(sid) < 6:
            sid = sid.zfill(6)
        picks.append(sid)
        weights.append(float(row['weight']))
    return picks, np.array(weights)


def load_stock_data():
    df = pd.read_csv(STOCK_DATA_PATH)
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    return df


def main():
    picks, weights = load_picks()
    stock_df = load_stock_data()

    print(f'Picks:  {picks}')
    print(f'Weights: {weights}')
    print()

    # locate trading days 6.29~7.3
    all_dates = sorted(stock_df['日期'].unique())
    dates = [d for d in all_dates if '2026-06-29' <= d <= '2026-07-03']
    if not dates:
        print('No data in range 2026-06-29 ~ 2026-07-03')
        return

    entry_date = dates[0]

    # for each stock, get entry open & daily close/涨跌幅
    daily_rets = []          # list of dicts, one per date
    entry_open = {}          # open price on entry_date

    for date in dates:
        day = stock_df[stock_df['日期'] == date]
        row = {}
        for c in picks:
            r = day[day['股票代码'] == c]
            row[c] = {
                'open': r['开盘'].values[0] if not r.empty else None,
                'close': r['收盘'].values[0] if not r.empty else None,
                'pct': r['涨跌幅'].values[0] if not r.empty else None,
            }
            if date == entry_date:
                entry_open[c] = row[c]['open']
        daily_rets.append({'date': date, 'row': row})

    # compute returns
    header_date = f'{"日期":<12}'
    header_picks = ''.join(f'{c:<12}' for c in picks)
    header_port = f'{"组合收益%":<12} {"累计收益%":<12}'
    print(header_date + header_picks + header_port)
    print('-' * (len(header_date) + len(header_picks) + len(header_port)))

    cum_val = np.array([1.0] * len(picks))

    for dr in daily_rets:
        date = dr['date']
        row = dr['row']
        rets = []
        for i, c in enumerate(picks):
            r = row[c]
            if r['close'] is None:
                rets.append(None)
                continue
            if date == entry_date:
                # day 1: (close - open) / open
                ret = (r['close'] - r['open']) / r['open'] * 100
            else:
                # subsequent days: close-to-close 涨跌幅
                ret = r['pct']
            rets.append(ret)

        # portfolio daily return (weighted)
        valid = [(ret, w) for ret, w in zip(rets, weights) if ret is not None]
        if valid:
            port_ret = sum(ret * w for ret, w in valid)
            # update cumulative (per-stock)
            for i, (ret, w) in enumerate(zip(rets, weights)):
                if ret is not None:
                    cum_val[i] *= (1 + ret / 100)
            # portfolio cumulative from weighted cum_val
            # total_value = sum(weight_i * cum_val_i)
            # cum_ret = (total_value - 1) * 100  (since initial total = sum weights = 1)
            total_val = np.sum(weights * cum_val)
            cum_ret = (total_val - 1) * 100
        else:
            port_ret = None
            cum_ret = None

        # print row
        date_str = f'{date:<12}'
        ret_strs = ''.join(f'{ret:<+11.4f} ' if ret is not None else f'{"N/A":<12}' for ret in rets)
        port_str = f'{port_ret:<+11.4f} ' if port_ret is not None else f'{"N/A":<12}'
        cum_str = f'{cum_ret:<+11.4f} ' if cum_ret is not None else f'{"N/A":<12}'
        print(date_str + ret_strs + port_str + cum_str)

    # final summary
    total_val = np.sum(weights * cum_val)
    final_cum = (total_val - 1) * 100
    print(f'\nFinal cumulative return (from {entry_date} open): {final_cum:+.4f}%')

    # per-stock cumulative from entry
    print(f'\nPer-stock cumulative (from {entry_date} open):')
    for c, v, w in zip(picks, cum_val, weights):
        print(f'  {c}: {(v-1)*100:+.4f}% (weight {w})')


if __name__ == '__main__':
    main()
