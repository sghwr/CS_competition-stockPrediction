import baostock as bs
import pandas as pd
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_PATH = 'data/index_data.csv'
START_DATE = "2019-01-01"


def login():
    lg = bs.login()
    if lg.error_code != '0':
        raise ConnectionError(f"baostock login failed: {lg.error_msg}")
    print("baostock登录成功")
    return lg


def logout():
    bs.logout()
    print("baostock已登出")


def fetch_index_data(start_date, end_date):
    """获取沪深300指数日线数据（后复权）"""
    rs = bs.query_history_k_data_plus(
        "sh.000300",
        "date,open,high,low,close,preclose,volume,amount,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="1"  # 后复权
    )
    rows = []
    while rs.next():
        row = rs.get_row_data()
        if row[0] is not None and row[0] != '':
            rows.append(row)

    if not rows:
        return None

    columns = ['date', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg']
    df = pd.DataFrame(rows, columns=columns)
    for col in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def add_derived_features(df):
    """添加衍生指标"""
    df['return_1d'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()
    return df


def load_existing_index_data(path):
    """加载已有指数数据，返回DataFrame或None"""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        print(f"  已加载现有指数数据: {len(df)} 条, {df['date'].min()} ~ {df['date'].max()}")
        return df
    except Exception as e:
        print(f"  警告: 读取现有指数数据失败: {e}")
        return None


def main():
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"目标数据时间范围: {START_DATE} 至 {end_date}")
    print(f"输出文件: {OUTPUT_PATH}")
    print("=" * 60)

    login()
    try:
        existing_df = load_existing_index_data(OUTPUT_PATH)

        if existing_df is not None and len(existing_df) > 0:
            existing_min = existing_df['date'].min()
            existing_max = existing_df['date'].max()
            need_early = existing_min > START_DATE
            need_late = existing_max < end_date

            if not need_early and not need_late:
                print(f"指数数据已完整 ({existing_min} ~ {existing_max})，跳过")
                return

            print(f"指数数据有缺口 (现有 {existing_min} ~ {existing_max})，将全量重新下载以保证后复权一致性")
        else:
            print("无现有指数数据，将全量下载")

        # 全量下载
        new_df = fetch_index_data(START_DATE, end_date)

        if new_df is None or new_df.empty:
            print("未获取到指数数据")
            return

        # 添加衍生指标
        new_df = add_derived_features(new_df)

        # 保存
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        new_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

        print(f"沪深300指数数据:")
        print(f"  数据量: {new_df.shape}")
        print(f"  日期范围: {new_df['date'].iloc[0]} ~ {new_df['date'].iloc[-1]}")
        print(f"  列: {new_df.columns.tolist()}")
        print(f"已保存至 {OUTPUT_PATH}")

    finally:
        logout()


if __name__ == '__main__':
    main()
