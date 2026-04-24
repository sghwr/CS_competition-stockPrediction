import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

OUTPUT_PATH = 'data/stock_industry.csv'

def get_trade_dates():
    lg = bs.login()
    if lg.error_code != '0':
        raise ConnectionError(f"baostock login failed: {lg.error_msg}")
    try:
        rs = bs.query_trade_dates(start_date="2019-01-01", end_date=datetime.now().strftime("%Y-%m-%d"))
        dates = []
        while rs.next():
            d = rs.get_row_data()
            if d[1] == '1':
                dates.append(d[0])
        return dates
    finally:
        bs.logout()

def fetch_industry_for_stock(code, date):
    rs = bs.query_stock_industry(code=code, date=date)
    while rs.next():
        row = rs.get_row_data()
        return {'code': row[1], 'industry': row[3], 'industry_sw': row[4]}
    return None

def main():
    trade_dates = get_trade_dates()
    if not trade_dates:
        print("No trade dates available")
        return

    hs300 = pd.read_csv('data/hs300_stock_list.csv')
    stock_codes = hs300['code'].tolist() if 'code' in hs300.columns else hs300.iloc[:, 0].tolist()
    stock_codes = [c.strip() for c in stock_codes if str(c).strip()]
    print(f"Total stocks in CSI300 pool: {len(stock_codes)}")

    mid_date = trade_dates[len(trade_dates) // 2]

    lg = bs.login()
    if lg.error_code != '0':
        raise ConnectionError(f"baostock login failed: {lg.error_msg}")

    records = []
    failed = []
    try:
        for i, code in enumerate(stock_codes):
            full_code = code
            info = fetch_industry_for_stock(full_code, mid_date)
            if info and info['industry'] and info['industry'] != '':
                records.append(info)
            elif info and info['industry_sw'] and info['industry_sw'] != '':
                info['industry'] = info['industry_sw']
                records.append(info)
            else:
                failed.append(code)
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(stock_codes)}")

    finally:
        bs.logout()

    df = pd.DataFrame(records)
    if failed:
        print(f"Failed to fetch industry for {len(failed)} stocks, assigning 'Unknown'")
        df_fallback = pd.DataFrame({'code': failed, 'industry': 'Unknown', 'industry_sw': 'Unknown'})
        df = pd.concat([df, df_fallback], ignore_index=True)

    print(f"Industry data shape: {df.shape}")
    print(f"Unique industries: {df['industry'].nunique()}")
    print(f"\nTop industries by count:\n{df['industry'].value_counts().head(15)}")

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
