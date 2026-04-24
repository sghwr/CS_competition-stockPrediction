"""
LGBM滑动窗口收益率回测评估脚本
==========================
在历史数据上逐日滑动，模拟 LGBM 模型预测 -> 选股 -> 持仓 -> 收益全过程。
独立脚本，仅需修改顶部配置区即可复用。

使用方法
-------
1. 修改下方 DATA_DIR / MODEL_DIR / TEST_CSV
2. python eval_lgbm.py
3. 查看 OUTPUT_DIR 下的 CSV 结果

所需文件
--------
DATA_DIR/stock_data.csv         # 日线行情数据
DATA_DIR/test.csv               # 评估日期来源
MODEL_DIR/lgbm_model.pkl        # 训练好的 LGBM 模型

依赖
----
numpy, pandas, joblib, tqdm, talib, lightgbm
"""

import os
import sys
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

# ============================= 配置区 =============================
DATA_DIR        = './data'                 # 数据文件夹
MODEL_DIR       = './LGBM/model'           # LGBM 模型路径
OUTPUT_DIR      = './output/eval_lgbm'     # 结果输出目录

TEST_CSV        = './data/test.csv'        # 评估日期来源

SEQUENCE_LENGTH = 60                       # 特征回溯窗口（必须匹配训练时）
FEATURE_NUM     = '158+39'                 # 特征版本 '39' | '158+39'
TOP_K           = 5                        # 每期选股数量
HOLDING_DAYS    = 5                        # 持有期（自然交易日）
# ================================================================

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils import engineer_features_39, engineer_features_158plus39

_FEATURE_ENGINEER = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
}

_FEATURE_COLUMNS = {
    '39': [
        '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread',
    ],
    '158+39': [
        '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2',
        'OPEN0', 'HIGH0', 'LOW0', 'VWAP0',
        'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60',
        'MA5', 'MA10', 'MA20', 'MA30', 'MA60',
        'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
        'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60',
        'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60',
        'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60',
        'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60',
        'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60',
        'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60',
        'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60',
        'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60',
        'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60',
        'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
        'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60',
        'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
        'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60',
        'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
        'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60',
        'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
        'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60',
        'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
        'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60',
        'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
        'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60',
        'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60',
        'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60',
        'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60',
        'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60',
        'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread',
    ],
}


# ======================== 辅助函数 ========================

def preprocess_data(df):
    """特征工程，返回 processed + feature_cols（无 scaler 不需要标准化）"""
    func = _FEATURE_ENGINEER[FEATURE_NUM]
    feature_cols = _FEATURE_COLUMNS[FEATURE_NUM]
    df = df.copy().sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    if not groups:
        raise ValueError('输入数据为空')
    nproc = min(8, mp.cpu_count())
    with mp.Pool(processes=nproc) as pool:
        processed_list = list(tqdm(
            pool.imap(func, groups), total=len(groups), desc='特征工程'
        ))
    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['日期'] = pd.to_datetime(processed['日期'])
    return processed, feature_cols


def get_holding_return(stock_code, pred_date, raw_df, hold=5):
    """持有期收益率 (score_self 风格): (最后开盘 - 最先开盘) / 最先开盘"""
    future = raw_df[
        (raw_df['股票代码'] == stock_code) & (raw_df['日期'] > pred_date)
    ].sort_values('日期').head(hold)
    if len(future) < hold:
        return None
    first_open = future.iloc[0]['开盘']
    last_open = future.iloc[-1]['开盘']
    return (last_open - first_open) / first_open


def compute_statistics(returns):
    """计算统计指标"""
    returns = np.array(returns, dtype=np.float64)
    n = len(returns)
    if n == 0:
        return {'总窗口数': 0}
    avg = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    cum = float(np.prod(1 + returns) - 1)
    sharpe = float(np.sqrt(252) * avg / std) if std > 0 else 0.0
    win = float(np.mean(returns > 0))
    pos = returns[returns > 0]
    neg = returns[returns < 0]
    pl = float(pos.mean() / abs(neg.mean())) if len(neg) and neg.mean() != 0 else float('inf')
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    mdd = float(np.min(drawdown))
    return {
        '总窗口数': n,
        '平均收益率': round(avg, 6),
        '收益率标准差': round(std, 6),
        '年化Sharpe': round(sharpe, 4),
        '胜率': round(win, 4),
        '盈亏比': round(pl, 4),
        '最大回撤': round(mdd, 6),
        '累计收益率': round(cum, 6),
        '最大单期收益': round(float(np.max(returns)), 6),
        '最小单期收益': round(float(np.min(returns)), 6),
        '中位数收益': round(float(np.median(returns)), 6),
    }


# ======================== 主流程 ========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_csv = os.path.abspath(os.path.join(DATA_DIR, 'stock_data.csv'))
    model_path = os.path.abspath(os.path.join(MODEL_DIR, 'lgbm_model.pkl'))

    for p in [data_csv, model_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'未找到: {p}')

    # ---------- 1. 加载原始数据 ----------
    print('>>> 加载原始数据 ...')
    raw_df = pd.read_csv(data_csv, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.strip()
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    stock_ids = sorted(raw_df['股票代码'].unique())
    print(f'    共 {len(stock_ids)} 只股票, {len(raw_df)} 行')

    # ---------- 2. 特征工程 ----------
    print('>>> 特征工程 ...')
    processed, features = preprocess_data(raw_df)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f'    特征维度: {len(features)}')

    # ---------- 3. 加载 LGBM 模型 ----------
    print('>>> 加载 LGBM 模型 ...')
    model = joblib.load(model_path)
    print(f'    模型类型: {type(model).__name__}')

    # ---------- 4. 确定可评估日期 ----------
    all_dates = sorted(processed['日期'].unique())
    min_date = all_dates[SEQUENCE_LENGTH - 1]
    max_date = all_dates[-1]

    if TEST_CSV:
        test_df = pd.read_csv(TEST_CSV)
        test_dates = sorted(pd.to_datetime(test_df['日期'].unique()))
        min_date = max(min_date, test_dates[0])
        max_date = min(max_date, test_dates[-1])
        print(f'    test.csv 日期范围: {test_dates[0].date()} ~ {test_dates[-1].date()}')

    eval_dates = [d for d in all_dates if min_date <= d <= max_date]
    print(f'>>> 评估日期: {min_date.date()} ~ {max_date.date()}, 共 {len(eval_dates)} 天')

    # ---------- 5. 逐日滑动评估 ----------
    results = []
    pbar = tqdm(eval_dates, desc='LGBM 滑动评估')
    for pred_date in pbar:
        date_str = pred_date.strftime('%Y-%m-%d')

        # --- 提取当日所有股票的特征（LGBM 格式：扁平向量 [N, F]） ---
        day_data = processed[processed['日期'] == pred_date]
        if len(day_data) < TOP_K:
            continue
        day_features = day_data[features].values.astype(np.float32)
        day_stock_codes = day_data['股票代码'].values

        # --- LGBM 推理 ---
        scores = model.predict(day_features)

        # --- 选股等权 ---
        topk_idx = np.argsort(scores)[::-1][:TOP_K]
        topk_ids = day_stock_codes[topk_idx]
        weights = np.array([1.0 / TOP_K] * TOP_K)

        # --- 计算加权收益率 ---
        rets, valid_sids, valid_ws = [], [], []
        for sid, w in zip(topk_ids, weights):
            r = get_holding_return(sid, pred_date, raw_df, HOLDING_DAYS)
            if r is not None:
                rets.append(r)
                valid_sids.append(sid)
                valid_ws.append(w)
        if not rets:
            continue
        rets, valid_ws = np.array(rets), np.array(valid_ws)
        valid_ws /= valid_ws.sum()
        window_ret = float(np.sum(rets * valid_ws))

        # --- 填充输出行 ---
        row = {'日期': date_str}
        for k in range(TOP_K):
            row[f'stock_{k+1}'] = valid_sids[k] if k < len(valid_sids) else ''
            row[f'weight_{k+1}'] = round(valid_ws[k], 4) if k < len(valid_ws) else 0.0
        row['加权收益率'] = round(window_ret, 6)
        results.append(row)
        pbar.set_postfix({'收益': f'{window_ret:.4%}'})

    if not results:
        print('未产生任何有效窗口，请检查数据或参数')
        return

    # ---------- 6. 结果输出 ----------
    result_df = pd.DataFrame(results)
    stats = compute_statistics(result_df['加权收益率'].values)
    stat_items = list(stats.items())
    stat_pairs = []
    for i in range(0, len(stat_items), 2):
        k1, v1 = stat_items[i]
        k2, v2 = stat_items[i + 1] if i + 1 < len(stat_items) else ('', '')
        stat_pairs.append((f'{k1}: {v1}', f'{k2}: {v2}'))

    stats_row_data = {'日期': '===== 统计汇总 ====='}
    for k in range(TOP_K):
        if k < len(stat_pairs):
            stats_row_data[f'stock_{k+1}'] = stat_pairs[k][0]
            stats_row_data[f'weight_{k+1}'] = stat_pairs[k][1]
        else:
            stats_row_data[f'stock_{k+1}'] = ''
            stats_row_data[f'weight_{k+1}'] = ''
    stats_row_data['加权收益率'] = ''

    stats_row = pd.DataFrame([stats_row_data])
    final_df = pd.concat([result_df, stats_row], ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, 'lgbm_results.csv')
    final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f'\n>>> 结果已保存: {csv_path}')
    print('\n======== LGBM 统计汇总 ========')
    for k, v in stats.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    mp.freeze_support()
    main()
