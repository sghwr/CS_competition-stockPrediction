"""
滑动窗口收益率回测评估脚本
==========================
在历史数据上逐日滑动，模拟模型预测 -> 选股 -> 持仓 -> 收益全过程。
每个窗口输出一次加权收益率，最终汇总全部窗口的统计指标。

使用方法
-------
1. 修改下方 DATA_DIR / MODEL_DIR / WINDOW_SIZE（5 ~ 365）
2. python eval_sliding_window.py
3. 查看 OUTPUT_DIR 下的 CSV 结果

所需文件
--------
DATA_DIR/stock_data.csv         # 日线行情数据（必需）
DATA_DIR/index_data.csv         # 指数数据（USE_MDRP=True 时需要）
DATA_DIR/stock_industry.csv     # 行业分类（USE_MDRP=True 时需要）
MODEL_DIR/best_model.pth        # 训练好的模型权重
MODEL_DIR/scaler.pkl            # 标准化器

依赖
----
numpy, pandas, torch, joblib, tqdm, talib
"""

import os
import sys
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm

# ============================= 配置区 =============================
DATA_DIR        = './data'                 # 数据文件夹
MODEL_DIR       = './model/60_158+39'      # 模型 + scaler 文件夹
OUTPUT_DIR      = './output/eval'          # 结果输出目录

WINDOW_SIZE     = 1                        # 滑动步长（交易日），1=逐日
TEST_CSV        = './data/test.csv'        # 留空=全量；填路径则仅评估 test.csv 内的日期

SEQUENCE_LENGTH = 60                       # 模型输入序列长度（必须匹配训练时）
FEATURE_NUM     = '158+39'                 # 特征版本 '39' | '158+39'
TOP_K           = 5                        # 每期选股数量
HOLDING_DAYS    = 5                        # 持有期（自然交易日）

USE_MDRP        = True                     # 启用 MD-SRP 先验
MDRP_LOOKBACK   = 5                        # 行业动量回溯窗口（交易日）

PORTFOLIO_TEMPERATURE = 0.5                # Softmax 温度
MAX_WEIGHT_PER_STOCK  = 0.4                # 单只股票最大权重

# ---------- 以下仅在训练时修改了模型结构时需要调整 ----------
MODEL_D_MODEL         = 256
MODEL_NHEAD           = 4
MODEL_NUM_LAYERS      = 3
MODEL_DIM_FEEDFORWARD = 512
MODEL_DROPOUT         = 0.1
# ================================================================

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from model import StockTransformer
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

def auto_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def preprocess_data(df, stockid2idx):
    """特征工程 + instrument 映射"""
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
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])
    return processed, feature_cols


def build_sequences(data, features, seq_len, stock_ids, date):
    """为指定日期构建推理序列 [N, seq_len, F]"""
    sequences, out_ids = [], []
    for sid in stock_ids:
        hist = data[(data['股票代码'] == sid) & (data['日期'] <= date)] \
            .sort_values('日期').tail(seq_len)
        if len(hist) == seq_len:
            sequences.append(hist[features].values.astype(np.float32))
            out_ids.append(sid)
    if not sequences:
        raise ValueError(f'{date.date()} 无足够历史数据的股票')
    return np.asarray(sequences, dtype=np.float32), out_ids


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
    device = auto_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_csv = os.path.abspath(os.path.join(DATA_DIR, 'stock_data.csv'))
    index_csv = os.path.abspath(os.path.join(DATA_DIR, 'index_data.csv'))
    industry_csv = os.path.abspath(os.path.join(DATA_DIR, 'stock_industry.csv'))
    model_path = os.path.abspath(os.path.join(MODEL_DIR, 'best_model.pth'))
    scaler_path = os.path.abspath(os.path.join(MODEL_DIR, 'scaler.pkl'))

    for p in [data_csv, model_path, scaler_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'未找到: {p}')

    # ---------- 1. 加载原始数据 ----------
    print('>>> 加载原始数据 ...')
    raw_df = pd.read_csv(data_csv, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.strip()
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: i for i, sid in enumerate(stock_ids)}
    print(f'    共 {len(stock_ids)} 只股票, {len(raw_df)} 行')

    # ---------- 2. 特征工程 ----------
    print('>>> 特征工程 ...')
    processed, features = preprocess_data(raw_df, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ---------- 3. 标准化 ----------
    print('>>> 加载 scaler 并标准化 ...')
    scaler = joblib.load(scaler_path)
    processed[features] = scaler.transform(processed[features])

    # ---------- 4. 构建模型 ----------
    print('>>> 加载模型 ...')
    model_cfg = {
        'sequence_length': SEQUENCE_LENGTH,
        'd_model': MODEL_D_MODEL,
        'nhead': MODEL_NHEAD,
        'num_layers': MODEL_NUM_LAYERS,
        'dim_feedforward': MODEL_DIM_FEEDFORWARD,
        'dropout': MODEL_DROPOUT,
    }
    model = StockTransformer(
        input_dim=len(features),
        config=model_cfg,
        num_stocks=len(stock_ids),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ---------- 5. MD-SRP 可选模块 ----------
    prior_computer = None
    state_extractor = None
    mdrp_actual = USE_MDRP
    if USE_MDRP:
        print('>>> 初始化 MD-SRP ...')
        from market_prior import IndustryPriorComputer, MarketStateExtractor
        if os.path.exists(industry_csv):
            prior_computer = IndustryPriorComputer(
                industry_csv=industry_csv,
                stock_data_csv=data_csv,
                lookback=MDRP_LOOKBACK,
            )
            print('    IndustryPriorComputer 就绪')
        else:
            print('    [跳过] stock_industry.csv 不存在')
        if os.path.exists(index_csv):
            state_extractor = MarketStateExtractor(index_csv=index_csv)
            print('    MarketStateExtractor 就绪')
        else:
            print('    [跳过] index_data.csv 不存在')
        if prior_computer is None and state_extractor is None:
            mdrp_actual = False

    # ---------- 6. 确定可评估日期 ----------
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
    eval_dates = eval_dates[::WINDOW_SIZE]
    print(f'>>> 评估日期: {min_date.date()} ~ {max_date.date()}, 步长={WINDOW_SIZE}天, 共 {len(eval_dates)} 个窗口')

    # ---------- 7. 逐日滑动评估 ----------
    results = []
    pbar = tqdm(eval_dates, desc='滑动评估')
    for pred_date in pbar:
        try:
            seq_np, seq_ids = build_sequences(
                processed, features, SEQUENCE_LENGTH, stock_ids, pred_date
            )
        except ValueError:
            continue

        date_str = pred_date.strftime('%Y-%m-%d')

        # --- 推理 ---
        with torch.no_grad():
            x = torch.from_numpy(seq_np).unsqueeze(0).to(device)
            pb = ms = None
            if mdrp_actual:
                if prior_computer is not None:
                    priors = prior_computer.get_prior_returns(date_str, seq_ids)
                    pb = torch.from_numpy(priors).float().to(device).unsqueeze(0)
                if state_extractor is not None:
                    oh = state_extractor.get_state_onehot(date_str)
                    vol = state_extractor.get_volatility(date_str)
                    ms_arr = np.concatenate([oh, [vol]]).astype(np.float32)
                    ms = torch.from_numpy(ms_arr).float().to(device).unsqueeze(0)
            scores = model(x, prior_bias=pb, market_state=ms).squeeze(0).cpu().numpy()

        # --- 选股等权 ---
        topk_idx = np.argsort(scores)[::-1][:TOP_K]
        topk_ids = [seq_ids[i] for i in topk_idx]
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

        # --- 填充 TOP_K 行宽输出 ---
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

    # ---------- 8. 结果输出 ----------
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
    # 剩余字段
    stats_row_data['加权收益率'] = ''

    stats_row = pd.DataFrame([stats_row_data])
    final_df = pd.concat([result_df, stats_row], ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, 'sliding_window_results.csv')
    final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f'\n>>> 结果已保存: {csv_path}')
    print('\n======== 统计汇总 ========')
    for k, v in stats.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    mp.freeze_support()
    main()
