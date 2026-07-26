#!/usr/bin/env python3
"""
获取沪深300指数成分股历史数据
- 获取2026年2月20日沪深300的300个成分股
- 抓取每只股票从2015年至今的历史量价数据
- 使用baostock平台
- 保存格式: 股票代码,日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌额,换手率,涨跌幅
"""

import baostock as bs
import pandas as pd
from datetime import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


def login():
    """登录baostock（支持重连，先清理可能残留的旧连接）"""
    try:
        bs.logout()
    except Exception:
        pass  # 旧连接已断开，忽略错误
    time.sleep(1)
    for attempt in range(3):
        lg = bs.login()
        if lg.error_code == '0':
            print("baostock登录成功")
            return lg
        wait = 3 * (attempt + 1)
        print(f"  登录失败({lg.error_msg})，{wait}秒后重试...")
        time.sleep(wait)
    raise Exception(f"登录失败: {lg.error_msg}")


def logout():
    """登出baostock"""
    bs.logout()
    print("baostock已登出")


def get_hs300_stocks():
    """获取沪深300成分股列表"""
    print("正在获取沪深300成分股列表...")
    
    rs = bs.query_hs300_stocks()
    
    if rs.error_code != '0':
        raise Exception(f"获取成分股失败: {rs.error_msg}")
    
    stocks = []
    while (rs.error_code == '0') & rs.next():
        stocks.append(rs.get_row_data())
    
    df = pd.DataFrame(stocks, columns=rs.fields)
    print(f"获取到 {len(df)} 只沪深300成分股")
    return df


def get_stock_history(bs_code, start_date, end_date):
    """获取单只股票历史数据"""
    rs = bs.query_history_k_data_plus(bs_code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="1")  # adjustflag="1"表示后复权
    
    if rs.error_code != '0':
        raise Exception(f"查询失败: {rs.error_msg}")
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    if not data_list:
        return None
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 转换数据类型
    numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 计算振幅和涨跌额
    df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(2)
    df['涨跌额'] = (df['close'] - df['preclose']).round(2)
    
    # 转换日期格式 YYYY/M/D
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # 提取纯数字股票代码（统一为6位格式，不足前面补0）
    df['code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
    df['code'] = df['code'].str.zfill(6)
    
    # 重命名列
    df = df.rename(columns={
        'code': '股票代码',
        'date': '日期',
        'open': '开盘',
        'close': '收盘',
        'high': '最高',
        'low': '最低',
        'volume': '成交量',
        'amount': '成交额',
        'turn': '换手率',
        'pctChg': '涨跌幅'
    })
    
    columns = ['股票代码', '日期', '开盘', '收盘', '最高', '最低', 
               '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅']
    df = df[columns]
    return df


def fetch_with_retry(bs_code, start_date, end_date, max_retries=5, base_delay=3, timeout=60):
    """带超时、重试和重连的股票数据获取"""
    for attempt in range(1, max_retries + 1):
        try:
            # 用线程池实现超时控制，防止baostock API永久挂起
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_stock_history, bs_code, start_date, end_date)
                try:
                    return future.result(timeout=timeout)
                except FuturesTimeoutError:
                    print(f"  [TIMEOUT {attempt}/{max_retries}] 查询超时({timeout}秒)，重连...")
                    login()
                    continue
        except Exception as e:
            err_str = str(e)
            # 网络断开类错误需要重连
            if '10054' in err_str or '10053' in err_str or '连接' in err_str or 'login' in err_str.lower():
                wait = base_delay * attempt
                print(f"  [RETRY {attempt}/{max_retries}] 连接断开，{wait}秒后重连...")
                time.sleep(wait)
                login()
                continue
            # 其他查询错误也重试，但等待更短
            wait = base_delay * attempt
            print(f"  [RETRY {attempt}/{max_retries}] {err_str}，{wait}秒后重试...")
            time.sleep(wait)
    print(f"  [FAIL] 重试 {max_retries} 次仍失败")
    return None


def build_stock_date_map(df, start_date=None, end_date=None):
    """一次性构建所有股票的日期范围映射，避免重复读文件"""
    if df is None or len(df) == 0:
        return {}
    df = df.copy()
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期_dt'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期_dt'])
    if start_date is not None:
        df = df[df['日期_dt'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df['日期_dt'] <= pd.to_datetime(end_date)]
    date_map = {}
    for code, group in df.groupby('股票代码'):
        date_map[code] = (group['日期_dt'].min().strftime('%Y-%m-%d'),
                          group['日期_dt'].max().strftime('%Y-%m-%d'))
    return date_map


def get_trade_calendar(start_date, end_date):
    """从baostock获取交易日历"""
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    if rs.error_code != '0':
        raise Exception(f"获取交易日历失败: {rs.error_msg}")
    trade_dates = []
    while (rs.error_code == '0') & rs.next():
        d = rs.get_row_data()
        if d[1] == '1':
            trade_dates.append(d[0])
    return sorted(trade_dates)


def check_data_gaps(df, trade_dates):
    """对比交易日历，检测并报告数据缺口"""
    if df is None or len(df) == 0:
        print("  无数据可供检查")
        return
    data_dates = set(df['日期'].astype(str).str[:10].unique())
    expected_dates = set(trade_dates)
    missing_dates = sorted(expected_dates - data_dates)
    if not missing_dates:
        print("  数据完整性检查通过: 无缺失交易日")
        return
    print(f"  ⚠ 警告: 发现 {len(missing_dates)} 个交易日数据缺失:")
    groups = []
    current_group = [missing_dates[0]]
    for i in range(1, len(missing_dates)):
        prev = datetime.strptime(missing_dates[i - 1], '%Y-%m-%d')
        curr = datetime.strptime(missing_dates[i], '%Y-%m-%d')
        if (curr - prev).days <= 7:
            current_group.append(missing_dates[i])
        else:
            groups.append(current_group)
            current_group = [missing_dates[i]]
    groups.append(current_group)
    for g in groups:
        if len(g) == 1:
            print(f"    {g[0]}")
        else:
            print(f"    {g[0]} ~ {g[-1]} ({len(g)}天)")


def _checkpoint_save(output_path, existing_df, all_new_data, stocks_to_replace):
    """定期存盘，将已收集的新数据与旧数据合并写入，防止中断丢失进度"""
    new_df = pd.concat(all_new_data, ignore_index=True)
    if existing_df is not None and len(existing_df) > 0:
        if stocks_to_replace:
            keep_mask = ~existing_df['股票代码'].astype(str).str.zfill(6).isin(stocks_to_replace)
            existing_df = existing_df[keep_mask]
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined['日期_dt'] = pd.to_datetime(combined['日期'], errors='coerce')
    combined = combined.sort_values(['股票代码', '日期_dt']).reset_index(drop=True)
    combined = combined.drop(columns=['日期_dt'])
    combined['日期'] = pd.to_datetime(combined['日期']).dt.strftime('%Y-%m-%d')
    combined.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  已存盘 {len(combined)} 行到 {output_path}")


def main():
    save_dir = os.path.abspath("./data")
    os.makedirs(save_dir, exist_ok=True)
    
    start_date = "2015-01-01"
    end_date = "2026-06-26"
    
    output_path = os.path.join(save_dir, "stock_data.csv")
    
    print(f"目标数据时间范围: {start_date} 至 {end_date}")
    print(f"输出文件: {output_path}")
    print("=" * 60)
    
    # 一次性读取已有数据
    existing_df = None
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path, dtype={'股票代码': str}, encoding='utf-8-sig')
            existing_df['股票代码'] = existing_df['股票代码'].astype(str).str.zfill(6)
            print(f"  已加载现有数据: {len(existing_df)} 条记录, {existing_df['股票代码'].nunique()} 只股票")
        except Exception as e:
            print(f"  警告: 读取现有数据失败: {e}")
            existing_df = None
    
    # 一次性构建日期范围映射
    date_map = build_stock_date_map(existing_df, start_date, end_date) if existing_df is not None else {}
    
    # 登录baostock
    login()
    
    try:
        # 获取交易日历用于缺口检测
        trade_calendar = get_trade_calendar(start_date, end_date)
        first_trade_date = trade_calendar[0] if trade_calendar else start_date
        print(f"  交易日历: {start_date} ~ {end_date} 共 {len(trade_calendar)} 个交易日 (首个交易日: {first_trade_date})")
        
        # 获取沪深300成分股
        hs300_df = get_hs300_stocks()
        
        # 保存成分股列表
        hs300_list_path = os.path.join(save_dir, "hs300_stock_list.csv")
        hs300_df.to_csv(hs300_list_path, index=False, encoding='utf-8-sig')
        
        # 准备处理所有股票
        hs300_df['纯代码'] = hs300_df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
        
        # 统计
        failed_stocks = []
        total = len(hs300_df)
        success_count = 0
        new_stock_count = 0
        replaced_count = 0
        total_new_records = 0
        all_new_data = []  # 收集所有新数据，最后一次性写入
        stocks_to_replace = set()  # 需要替换旧数据的股票代码（因后复权因子可能已变）
        
        for idx, row in hs300_df.iterrows():
            bs_code = row.get('code', '')
            stock_name = row.get('code_name', '')
            pure_code = row.get('纯代码', '')
            
            existing_range = date_map.get(pure_code)
            
            if existing_range:
                existing_min_date, existing_max_date = existing_range
                need_late = existing_max_date < end_date
                
                if not need_late:
                    print(f"[{idx+1}/{total}] {bs_code} {stock_name} - 已完整 ({existing_min_date}~{existing_max_date}) 跳过")
                    continue
                
                # 需要追加近期数据，全量重新下载以保证后复权一致性
                print(f"[{idx+1}/{total}] {bs_code} {stock_name} - 重新全量下载 (旧范围 {existing_min_date}~{existing_max_date})")
                stocks_to_replace.add(pure_code)
            else:
                print(f"[{idx+1}/{total}] {bs_code} {stock_name} - 全新获取")
            
            try:
                stock_data = fetch_with_retry(bs_code, start_date, end_date)
                
                if stock_data is not None and not stock_data.empty:
                    all_new_data.append(stock_data)
                    total_new_records += len(stock_data)
                    success_count += 1
                    if pure_code in stocks_to_replace:
                        replaced_count += 1
                        print(f"  [OK] 全量替换 {len(stock_data)}行 ({stock_data['日期'].iloc[0]}~{stock_data['日期'].iloc[-1]})")
                    else:
                        new_stock_count += 1
                        print(f"  [OK] +{len(stock_data)}行 ({stock_data['日期'].iloc[0]}~{stock_data['日期'].iloc[-1]})")
                else:
                    print(f"  [SKIP] 无数据")
                    failed_stocks.append((bs_code, stock_name))
                    
            except Exception as e:
                print(f"  [FAIL] {e}")
                failed_stocks.append((bs_code, stock_name))
            
            # 请求间隔，避免触发baostock服务端限流
            time.sleep(1)
            
            # 每50只股票存盘一次，防止中断丢失进度
            if (idx + 1) % 50 == 0 and all_new_data:
                print(f"  --- 存盘检查点 ({idx+1}/{total}) ---")
                _checkpoint_save(output_path, existing_df, all_new_data, stocks_to_replace)
        
        # === 最终合并并写入 ===
        print("\n" + "=" * 60)
        if all_new_data:
            new_df = pd.concat(all_new_data, ignore_index=True)
            if existing_df is not None and len(existing_df) > 0:
                # 移除被替换股票的旧数据（避免后复权不一致）
                if stocks_to_replace:
                    keep_mask = ~existing_df['股票代码'].astype(str).str.zfill(6).isin(stocks_to_replace)
                    existing_df = existing_df[keep_mask]
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined['日期_dt'] = pd.to_datetime(combined['日期'], errors='coerce')
                combined = combined.sort_values(['股票代码', '日期_dt']).reset_index(drop=True)
                combined = combined.drop(columns=['日期_dt'])
            else:
                combined = new_df
                combined['日期_dt'] = pd.to_datetime(combined['日期'], errors='coerce')
                combined = combined.sort_values(['股票代码', '日期_dt']).reset_index(drop=True)
                combined = combined.drop(columns=['日期_dt'])
            
            # 统一日期格式为 YYYY-MM-DD（避免历史数据与增量数据格式不一致）
            combined['日期'] = pd.to_datetime(combined['日期']).dt.strftime('%Y-%m-%d')
            combined.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"已写入 {len(combined)} 行到 {output_path}")
        else:
            print("无新数据需要写入")
        
        # 显示结果
        print(f"  - 全新获取: {new_stock_count} 只")
        print(f"  - 全量替换: {replaced_count} 只（后复权一致）")
        print(f"  - 已完整跳过: {total - success_count - len(failed_stocks)} 只")
        print(f"  - 失败: {len(failed_stocks)} 只")
        print(f"  - 新增记录: {total_new_records}")
        
        # 验证
        if os.path.exists(output_path):
            df = pd.read_csv(output_path, dtype={'股票代码': str}, encoding='utf-8-sig')
            print(f"\n文件总览: {len(df)} 行, {df['股票代码'].nunique()} 只股票")
            print(f"  日期范围: {df['日期'].min()} ~ {df['日期'].max()}")
            # 数据缺口检测
            print("\n--- 数据完整性检查 ---")
            check_data_gaps(df, trade_calendar)
        
        if failed_stocks:
            failed_df = pd.DataFrame(failed_stocks, columns=['股票代码', '股票名称'])
            failed_path = os.path.join(save_dir, "failed_stocks.csv")
            failed_df.to_csv(failed_path, index=False, encoding='utf-8-sig')
            print(f"\n失败股票列表已保存至: {failed_path}")
    
    finally:
        logout()


if __name__ == "__main__":
    main()
