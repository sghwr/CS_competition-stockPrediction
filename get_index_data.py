import baostock as bs
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUTPUT_PATH = 'data/index_data.csv'

def main():
    lg = bs.login()
    if lg.error_code != '0':
        raise ConnectionError(f"baostock login failed: {lg.error_msg}")
    try:
        rs = bs.query_history_k_data_plus(
            "sh.000300",
            "date,open,high,low,close,preclose,volume,amount,pctChg",
            start_date="2019-01-01",
            end_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2"
        )
        rows = []
        while rs.next():
            row = rs.get_row_data()
            if row[0] is not None and row[0] != '':
                rows.append(row)
        columns = ['date', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg']
        df = pd.DataFrame(rows, columns=columns)
        for col in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)
        df['return_1d'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        print(f"CSI300 index data shape: {df.shape}")
        print(f"Date range: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
        print(f"Columns: {df.columns.tolist()}")
        df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"Saved to {OUTPUT_PATH}")
    finally:
        bs.logout()

if __name__ == '__main__':
    main()
