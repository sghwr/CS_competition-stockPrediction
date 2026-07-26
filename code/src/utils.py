import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm
# torch 改为按需 import (避免被无关 worker 进程加载 1-2GB)

# 特征工程
def _rolling_linear_regression(x, y):
    x = np.vstack([np.ones(len(x)), x]).T
    beta, res, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta[1], res[0] if len(res) > 0 else 0.0, np.sum((y - (x @ beta))**2)
def engineer_features_158plus39(df):
    """
    计算39个技术指标特征和158个Alpha特征，并合并它们。
    """
    # 1. 计算158个Alpha特征（函数内部自行拷贝）
    df_158 = engineer_features(df)
    
    # 2. 计算39个技术指标特征（函数内部自行拷贝）
    df_39 = engineer_features_39(df)

    # 3. 合并两个DataFrame
    # 首先，从df_39中选取我们需要的列，避免与df_158中的原始列（如'开盘'）重复
    feature_cols_39 = [
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 
        'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 
        'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 
        'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ]
    
    # 确保所有列都存在于df_39中
    feature_cols_39_exist = [col for col in feature_cols_39 if col in df_39.columns]
    
    # 合并，df_158 已经包含了原始列和158个特征
    df_final = pd.concat([df_158, df_39[feature_cols_39_exist]], axis=1)

    # 4. 处理可能因为合并产生的重复列（如果两个函数生成了同名特征）
    df_final = df_final.loc[:,~df_final.columns.duplicated()]

    # 5. 统一处理inf和NaN
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)

    return df_final

# T+4 持仓专用核心因子 (52 维)
CORE_FEATURE_COLUMNS = [
    # 原始 10
    '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    # 短期反转 (8) - 散户超卖反弹
    'RANK10', 'RANK20', 'RSV10', 'RSV20', 'QTLD10', 'QTLD20', 'CNTD10', 'CNTD20',
    # 中期动量 (8) - 趋势惯性
    'ROC10', 'ROC20', 'BETA10', 'BETA20', 'RESI10', 'RESI20', 'SUMD10', 'SUMD20',
    # 量价信号 (9) - 主力资金意图
    'obv', 'CORR10', 'CORR20', 'VMA10', 'VMA20', 'VSTD10', 'VSTD20', 'VSUMD10', 'VSUMD20',
    # 波动率 (5) - 均值回归
    'STD10', 'STD20', 'volatility_20', 'atr_14', 'boll_std',
    # 趋势/均线 (3) - 趋势确认
    'MA10', 'MA20', 'MA60',
    # 技术指标 (4) - 经典情绪
    'rsi', 'macd', 'kdj_k', 'boll_mid',
    # 反转K线 (3) - 微观结构
    'KSFT', 'KUP2', 'KLOW2',
    # 收益率 (2) - 短期+中期
    'return_5', 'return_10',
]

def engineer_features_core(df):
    """
    T+4 持仓专用核心因子：从 196 维全特征中精选 52 维。
    复用 engineer_features_158plus39 的计算结果，只做列筛选。
    保留元数据列 (股票代码/日期) 供下游 _preprocess_common 使用。
    """
    df_full = engineer_features_158plus39(df)
    meta_cols = ['股票代码', '日期']
    selected = [c for c in CORE_FEATURE_COLUMNS if c in df_full.columns]
    return df_full[meta_cols + selected]


# ---------------------------------------------------------------------------
# 分层因子工程 (187 dim)
#   Tier 1 (97 维): 末值即可，无需衍生
#   Tier 2 (89 维): 末值
#   Tier 3 (1 维):  industry_id (categorical)
#   总量: 97 + 89 + 1 = 187
# ---------------------------------------------------------------------------

RAW_FEATURES_7 = ['开盘', '最高', '最低', '收盘', '成交量', '成交额', '换手率']

TIER1_LAST_ONLY = [
    # 9 K-line
    'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2',
    # 4 price ratio
    'OPEN0', 'HIGH0', 'LOW0', 'VWAP0',
    # 5 RANK
    'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60',
    # 5 RSV
    'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60',
    # 15 position
    'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
    'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60',
    'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
    # 15 count
    'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60',
    'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
    'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60',
    # 15 sum ratio
    'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
    'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60',
    'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
    # 15 volume sum ratio
    'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60',
    'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60',
    'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
    # 10 corr
    'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60',
    'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
    # 4 spread
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread',
]

TIER2_ENRICHED = [
    # 5 momentum
    'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60',
    # 5 MA
    'MA5', 'MA10', 'MA20', 'MA30', 'MA60',
    # 5 STD
    'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
    # 5 BETA
    'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60',
    # 5 RSQR
    'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60',
    # 5 RESI
    'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60',
    # 5 MAX
    'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60',
    # 5 MIN
    'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60',
    # 5 QTLU
    'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60',
    # 5 QTLD
    'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60',
    # 5 VMA
    'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60',
    # 5 VSTD
    'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60',
    # 5 WVMA
    'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60',
    # 39 technical
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
    'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
    'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60',
    'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
]


# Tier 3: 行业 categorical 特征 (1 维)
# 删 TIER3_INDUSTRY_RELATIVE (6) + TIER3_INDUSTRY_CONTEXT (3) = 9 dim
#   理由: industry features 跟 MacroTransformer 输出重复
# 删 TIER3_INDUSTRY_BETA (3) + TIER3_MACRO (5) = 8 dim
#   理由: 跟 MacroTransformer industry_beta + regime 重复
# 保留 TIER3_INDUSTRY_CATEGORICAL (1): industry_id (LightGBM categorical feature)
# Tree 总计: 97 Tier1 + 89 Tier2 + 1 industry_id = 187 dim
TIER3_INDUSTRY_CATEGORICAL = [
    'industry_id',     # 0..10 类别特征
]

TIER3_INDUSTRY = TIER3_INDUSTRY_CATEGORICAL
# Total: 1 维 (industry_id only)


def add_industry_beta_features(df, industry_map, lookback=60, min_valid=20):
    """为每只股票计算 60d 滚动 beta / alpha / 残差波动率 (相对其行业).

    公式 (O(T) 向量化, 用 cumsum):
      beta  = cov(stock_ret, ind_ret) / var(ind_ret)
      alpha = mean(stock_ret) - beta * mean(ind_ret)
      resid_std = std(stock_ret - alpha - beta * ind_ret)

    Args:
        df: 必须含 '日期', '股票代码', '收盘' 列
        industry_map: {code_int: industry_name}
        lookback: 滚动窗口 (默认 60)
        min_valid: 窗口内最少有效样本数 (默认 20)

    Returns:
        新的 df, 附加 'beta_60', 'alpha_60', 'resid_std_60', 'industry_id' 列
    """
    df = df.copy().sort_values(['股票代码', '日期']).reset_index(drop=True)

    # 1. industry_id
    if 'industry_id' not in df.columns:
        industries = sorted(set(industry_map.values()))
        industry_to_idx = {ind: i for i, ind in enumerate(industries)}
        df['industry_id'] = df['股票代码'].map(
            lambda c: industry_to_idx.get(industry_map.get(int(c), ''), 0)
        ).astype('int32')

    # 2. daily return
    df['ret_d'] = df.groupby('股票代码')['收盘'].pct_change()

    # 3. daily industry return (per date × industry 均值)
    df['ind_ret_d'] = df.groupby(['日期', 'industry_id'])['ret_d'].transform('mean')

    # 4. rolling beta/alpha/resid_std per stock (O(T) 向量化)
    betas, alphas, resid_stds = _rolling_beta_per_stock(
        df['股票代码'].values, df['ret_d'].values, df['ind_ret_d'].values,
        lookback=lookback, min_valid=min_valid,
    )
    df['beta_60'] = betas
    df['alpha_60'] = alphas
    df['resid_std_60'] = resid_stds

    # 清理中间列
    df = df.drop(columns=['ret_d', 'ind_ret_d'])
    return df


def _rolling_beta_per_stock(stock_ids, ret, ind_ret, lookback=60, min_valid=20):
    """对每只股票用 cumsum 算 O(T) 滚动 beta/alpha/resid_std."""
    from collections import defaultdict
    by_stock = defaultdict(list)
    for i, sid in enumerate(stock_ids):
        by_stock[sid].append(i)

    n = len(ret)
    betas = np.full(n, np.nan, dtype=np.float32)
    alphas = np.full(n, np.nan, dtype=np.float32)
    resid_stds = np.full(n, np.nan, dtype=np.float32)

    for sid, idxs in by_stock.items():
        x = ret[idxs]
        y = ind_ret[idxs]
        T = len(x)
        if T < lookback:
            continue

        # cumsum (NaN -> 0, count 单独算)
        x_filled = np.where(np.isnan(x), 0.0, x)
        y_filled = np.where(np.isnan(y), 0.0, y)
        valid = (~np.isnan(x)) & (~np.isnan(y))
        valid_f = valid.astype(np.float64)
        xy_filled = np.where(valid, x_filled * y_filled, 0.0)
        y2_filled = y_filled * y_filled

        Sx = np.cumsum(x_filled)
        Sy = np.cumsum(y_filled)
        Sxy = np.cumsum(xy_filled)
        Sy2 = np.cumsum(y2_filled)
        Sv = np.cumsum(valid_f)

        # window [i-lookback, i) 的 sum: cumsum[i-1] - cumsum[i-lookback-1]
        b_arr = np.full(T, np.nan, dtype=np.float32)
        a_arr = np.full(T, np.nan, dtype=np.float32)
        r_arr = np.full(T, np.nan, dtype=np.float32)

        for i in range(lookback, T):
            if i - lookback - 1 >= 0:
                sum_x = Sx[i-1] - Sx[i-lookback-1]
                sum_y = Sy[i-1] - Sy[i-lookback-1]
                sum_xy = Sxy[i-1] - Sxy[i-lookback-1]
                sum_y2 = Sy2[i-1] - Sy2[i-lookback-1]
                cnt = Sv[i-1] - Sv[i-lookback-1]
            else:
                sum_x = Sx[i-1]
                sum_y = Sy[i-1]
                sum_xy = Sxy[i-1]
                sum_y2 = Sy2[i-1]
                cnt = Sv[i-1]
            if cnt < min_valid:
                continue
            xm = sum_x / cnt
            ym = sum_y / cnt
            cov = sum_xy / cnt - xm * ym
            var = sum_y2 / cnt - ym * ym
            if var < 1e-9:
                continue
            b = cov / var
            a = xm - b * ym
            # resid_std 需要循环 O(lookback), 不可避免
            xw = x[i-lookback:i]
            yw = y[i-lookback:i]
            vmask = valid[i-lookback:i]
            if vmask.sum() > 0:
                xv, yv = xw[vmask], yw[vmask]
                rstd = (xv - a - b * yv).std()
            else:
                rstd = np.nan
            b_arr[i] = b
            a_arr[i] = a
            r_arr[i] = rstd

        for k, gi in enumerate(idxs):
            betas[gi] = b_arr[k]
            alphas[gi] = a_arr[k]
            resid_stds[gi] = r_arr[k]

    return betas, alphas, resid_stds


def _slope_20(series_tail_20):
    """20 天线性斜率 (归一化到相对量级).
    Args:
        series_tail_20: pd.Series, 长度 20
    Returns:
        float: slope / (|mean| + 1e-9)
    """
    arr = np.asarray(series_tail_20, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = arr.mean()
    num = np.sum((x - x_mean) * (arr - y_mean))
    den = np.sum((x - x_mean) ** 2) + 1e-12
    slope = num / den
    norm = np.abs(y_mean) + 1e-9
    val = slope / norm
    if not np.isfinite(val):
        return 0.0
    return float(val)


def enrich_window_factors(per_stock_window_df, mode='tiered'):
    """对单只股票 60d 窗口的衍生因子进行分层衍生。
    Args:
        per_stock_window_df: DataFrame, 单只股票最近 60 天的衍生因子 (60 × N)
            必须包含 TIER1_LAST_ONLY + TIER2_ENRICHED 中存在的列
        mode: 'tiered' (275) / 'snapshot_only' (186) / 'full' (373, 含 mean_20 + std_20)
    Returns:
        dict: {feature_name: value}
    """
    out = {}

    t1_cols = [c for c in TIER1_LAST_ONLY if c in per_stock_window_df.columns]
    for c in t1_cols:
        out[c] = per_stock_window_df[c].iloc[-1]

    t2_cols = [c for c in TIER2_ENRICHED if c in per_stock_window_df.columns]

    if mode == 'snapshot_only':
        for c in t2_cols:
            out[c] = per_stock_window_df[c].iloc[-1]
    elif mode == 'tiered':
        for c in t2_cols:
            out[c] = per_stock_window_df[c].iloc[-1]
            tail20 = per_stock_window_df[c].tail(20)
            if len(tail20) >= 2:
                out[f'{c}_slope20'] = _slope_20(tail20)
            else:
                out[f'{c}_slope20'] = 0.0
    elif mode == 'full':
        for c in t2_cols:
            tail = per_stock_window_df[c]
            out[c] = tail.iloc[-1]
            tail20 = tail.tail(20)
            if len(tail20) >= 2:
                out[f'{c}_mean20'] = tail20.mean()
                out[f'{c}_std20'] = tail20.std()
                out[f'{c}_slope20'] = _slope_20(tail20)
            else:
                out[f'{c}_mean20'] = 0.0
                out[f'{c}_std20'] = 0.0
                out[f'{c}_slope20'] = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return out

def engineer_features_39(df):
    """
    计算39个技术指标特征。
    'stock_idx','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
    'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 
    'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    """
    try:
        import talib
        import numpy as np
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    # 移动平均线 (SMA, EMA)
    df['sma_5'] = talib.SMA(close, timeperiod=5)
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['ema_26'] = talib.EMA(close, timeperiod=26)
    df['ema_60'] = talib.EMA(close, timeperiod=60)

    # MACD
    macd_line, macd_signal_line, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal_line

    # RSI
    df['rsi'] = talib.RSI(close, timeperiod=14)

    # KDJ
    df['kdj_k'], df['kdj_d'] = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # Bollinger Bands
    df['boll_mid'], df['boll_upper'], df['boll_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    # 标准差 = (上轨 - 中轨) / 2
    df['boll_std'] = (df['boll_upper'] - df['boll_mid']) / 2

    # 删除临时列
    df.drop(columns=['boll_upper', 'boll_lower'], inplace=True)

    # ATR
    df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

    # OBV (On-Balance Volume)
    df['obv'] = talib.OBV(close, volume)

    # Volume-related features
    df['volume_change'] = volume.pct_change(fill_method=None)
    df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
    df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = df['volume_ma_5'] / df['volume_ma_20']

    # Returns and Volatility
    df['return_1'] = close.pct_change(1)
    df['return_5'] = close.pct_change(5)
    df['return_10'] = close.pct_change(10)
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()

    # Spreads
    df['high_low_spread'] = high - low
    df['open_close_spread'] = open_ - close
    df['high_close_spread'] = high - close
    df['low_close_spread'] = low - close

    # 处理 inf 和 -inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 填充 NaN 值（注意：这可能引入偏差，根据下游任务决定是否保留）
    df.fillna(0, inplace=True)

    return df

def engineer_features(df):
    """
    使用talib加速特征计算
    """
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    # 为了避免修改原始DataFrame，创建一个副本
    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    vwap = df['成交额'] / (volume + 1e-12)

    # 特征列表
    features = []
    feature_names = []

    # 1. K-line features (9 features) - 向量化操作，速度很快，无需更改
    features.extend([
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12)
    ])
    feature_names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])

    # 2. Price-related features (4 features) - 向量化操作，无需更改
    features.extend([
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12)
    ])
    feature_names.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])

    windows = [5, 10, 20, 30, 60]

    # 3. Price change features (5 features) - 向量化操作，无需更改
    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f'ROC{w}')

    # 4. Moving average features (5 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MA{w}')

    # 5. Standard deviation features (5 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'STD{w}')

    # 6. Regression-based features (15 features) - 使用 talib 加速
    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f'BETA{w}')
        
        # R-squared can be calculated as CORREL^2
        time_period_series = pd.Series(
            np.tile(np.arange(w), len(close) // w + 1)[:len(close)],
            index=close.index
        )
        rolling_corr = close.rolling(w).corr(time_period_series)
        rsquare = rolling_corr**2
        features.append(rsquare)
        feature_names.append(f'RSQR{w}')

        # Residuals
        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
        predicted = slope * (w - 1) + intercept
        resi = close - predicted
        features.append(resi / (close + 1e-12))
        feature_names.append(f'RESI{w}')

    # 7. Max/Min features (10 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MAX{w}')
    for w in windows:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MIN{w}')

    # 8. Quantile features (10 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f'QTLU{w}')
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f'QTLD{w}')

    # 9. Rank features (5 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f'RANK{w}')

    # 10. Stochastic oscillator features (5 features) - talib.STOCH 计算的是另一指标，保留原实现
    for w in windows:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        feature_names.append(f'RSV{w}')

    # 11. Index of Max/Min features (15 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(high.rolling(w).apply(np.argmax, raw=True) / w)
        feature_names.append(f'IMAX{w}')
    for w in windows:
        features.append(low.rolling(w).apply(np.argmin, raw=True) / w)
        feature_names.append(f'IMIN{w}')
    for w in windows:
        imax = high.rolling(w).apply(np.argmax, raw=True)
        imin = low.rolling(w).apply(np.argmin, raw=True)
        features.append((imax - imin) / w)
        feature_names.append(f'IMXD{w}')

    # 12. Correlation features (10 features) - 使用 talib 加速
    log_volume = np.log(volume + 1)
    for w in windows:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        feature_names.append(f'CORR{w}')
    
    close_ret = close / close.shift(1)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    log_volume_ret = np.log(volume_ret + 1)
    for w in windows:
        # talib.CORREL 需要 Series，且不能有 NaN
        corr_df = pd.concat([close_ret, log_volume_ret], axis=1).fillna(0)
        features.append(talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w))
        feature_names.append(f'CORD{w}')

    # 13. Count features (15 features) - 向量化操作，无需更改
    close_diff_pos = (close > close.shift(1))
    close_diff_neg = (close < close.shift(1))
    for w in windows:
        features.append(close_diff_pos.rolling(w).mean())
        feature_names.append(f'CNTP{w}')
    for w in windows:
        features.append(close_diff_neg.rolling(w).mean())
        feature_names.append(f'CNTN{w}')
    for w in windows:
        cntp = close_diff_pos.rolling(w).mean()
        cntn = close_diff_neg.rolling(w).mean()
        features.append(cntp - cntn)
        feature_names.append(f'CNTD{w}')

    # 14. Sum of price change features (15 features) - 向量化操作，无需更改
    close_diff_abs = (close - close.shift(1)).abs()
    close_diff_up = (close - close.shift(1)).clip(lower=0)
    close_diff_down = -(close - close.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'SUMP{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'SUMN{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'SUMD{w}')

    # 15. Volume-related features (10 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VMA{w}')
    for w in windows:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VSTD{w}')

    # 16. Weighted volume features (5 features) - 向量化操作，无需更改
    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in windows:
        mean_vol_w_ret = vol_weighted_ret.rolling(w).mean()
        std_vol_w_ret = vol_weighted_ret.rolling(w).std()
        features.append(std_vol_w_ret / (mean_vol_w_ret + 1e-12))
        feature_names.append(f'WVMA{w}')

    # 17. Volume change sum features (15 features) - 向量化操作，无需更改
    volume_diff_abs = (volume - volume.shift(1)).abs()
    volume_diff_up = (volume - volume.shift(1)).clip(lower=0)
    volume_diff_down = -(volume - volume.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'VSUMP{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'VSUMN{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'VSUMD{w}')

    # Combine all features into a new DataFrame
    feature_df = pd.concat(features, axis=1)
    feature_df.columns = feature_names
    
    # Merge with original df
    df = pd.concat([df, feature_df], axis=1)
    
    # 填充缺失值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df
def process_single_stock(stock_row, data, features, sequence_length, date):
    """处理单只股票的数据，返回序列、目标值和股票索引"""
    stock_code = stock_row['instrument']
    # stock_idx = stock_row['stock_idx']
    
    # 获取该股票历史sequence_length天的数据（包括当天）
    stock_history = data[
        (data['instrument'] == stock_code) & 
        (data['datetime'] <= date)
    ].sort_values('datetime').tail(sequence_length)

    if len(stock_history) == sequence_length:
        seq = stock_history[features].values
        target = stock_row['label']  # 下一天的涨跌幅
        return seq, target, stock_code
    else:
        return None, None, None

def process_single_date(date, data, features, sequence_length):
    """处理单个日期的所有股票数据"""
    try:
        # 获取当天有target的股票（即有下一天数据的股票）
        day_data = data[data['datetime'] == date]
        day_data = day_data.dropna(subset=['label'])  # 确保有target
        
        if len(day_data) < 10:  # 确保至少有10只股票
            return None
            
        # 获取当天所有股票的特征序列
        day_sequences = []
        day_targets = []
        day_stock_indices = []
        
        # 对于单个日期内的股票处理，仍使用串行方式避免过度并行化
        # 因为多进程的开销可能超过收益
        for _, stock_row in day_data.iterrows():
            seq, target, stock_idx = process_single_stock(
                stock_row, data, features, sequence_length, date
            )
            if seq is not None:
                day_sequences.append(seq)
                day_targets.append(target)
                day_stock_indices.append(stock_idx)
        
        if len(day_sequences) >= 10:  # 确保有足够的股票
            # 创建排序标签：涨跌幅越高，相关性得分越高
            day_targets = np.array(day_targets)
            # 使用涨跌幅的排序作为相关性得分（值越大排名越高）
            sorted_indices = np.argsort(day_targets)[::-1]  # 降序排列
            relevance = np.zeros_like(day_targets, dtype=np.float32)
            for rank, idx in enumerate(sorted_indices):
                relevance[idx] = len(day_targets) - rank  # 最高涨跌幅得分最高
            
            return {
                'sequences': np.array(day_sequences),
                'targets': day_targets,
                'relevance': relevance,
                'stock_indices': day_stock_indices,
                'date': date
            }
        else:
            return None
            
    except Exception as e:
        print(f"处理日期 {date} 时出错: {e}")
        return None

def create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path=None, max_workers=None):
    """
    输入：股票历史数据 DataFrame，特征列名列表，序列长度，排名数据保存路径，最大工作进程数
    输出：排序数据集，格式为：(sequences, targets, relevance_scores, stock_indices)
    - sequences: List of np.array, 每个元素形状为 (num_stocks, sequence_length, num_features)
    - targets: List of np.array, 每个元素形状为 (num_stocks,)
    - relevance_scores: List of np.array, 每个元素形状为 (num_stocks,)
    - stock_indices: List of List, 每个元素为对应股票的索引列表
    """
    """多进程版本的排序数据集创建函数"""
    if ranking_data_path is not None:
        # 如果指定了ranking_data_path，尝试加载已有的数据集
        if os.path.exists(ranking_data_path):
            print(f"加载已有的排序数据集: {ranking_data_path}")
            return joblib.load(ranking_data_path)
    """
    创建排序数据集，按日期组织数据，每个样本包含同一天所有股票的特征和涨跌幅排序
    使用多线程加速处理
    """
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []
    
    print("正在创建排序数据集（多线程版本）...")
    
    # 获取所有日期，确保有足够的历史数据
    all_dates = sorted(data['datetime'].unique())
    min_date_for_sequences = all_dates[sequence_length-1]  # 确保有足够历史数据
    
    # 只使用有足够历史数据的日期
    valid_dates = [date for date in all_dates if date >= min_date_for_sequences]
    
    print(f"总日期数: {len(all_dates)}, 有效日期数: {len(valid_dates)}")
    
    # 设置最大工作进程数
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from tqdm import tqdm
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 10)
    
    print(f"使用 {max_workers} 个进程处理数据")
    
    # 分批处理日期以避免内存问题
    processed_count = 0
        
    # 使用进程池并行处理日期批次
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 创建处理函数的偏函数
            process_func = partial(process_single_date,
                                    data=data,
                                    features=features,
                                    sequence_length=sequence_length)
            
            # 并行处理批次中的所有日期
            futures = [executor.submit(process_func, date) for date in valid_dates]
            
            # 收集结果
            for future in tqdm(futures, desc="Processing dates", total=len(valid_dates)):
                try:
                    result = future.result(timeout=60)  # 设置超时
                    if result is not None:
                        sequences.append(result['sequences'])
                        targets.append(result['targets'])
                        relevance_scores.append(result['relevance'])
                        stock_indices.append(result['stock_indices'])
                        processed_count += 1
                except Exception as e:
                    print(f"处理某个日期时出错: {e}")
                    continue
                    
    except Exception as e:
        print(f"进程池处理出错，回退到串行处理: {e}")
        # 如果多进程出错，回退到串行处理
        for date in tqdm(valid_dates, desc="串行处理"):
            result = process_single_date(date, data, features, sequence_length)
            if result is not None:
                sequences.append(result['sequences'])
                targets.append(result['targets'])
                relevance_scores.append(result['relevance'])
                stock_indices.append(result['stock_indices'])
                processed_count += 1
    
    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        print(f"每个样本平均包含 {np.mean([len(seq) for seq in sequences]):.1f} 只股票")
    
    # 将四个数据保存下来，下次直接读取
    if ranking_data_path:
        joblib.dump((sequences, targets, relevance_scores, stock_indices), ranking_data_path)
        print(f"数据集已保存到: {ranking_data_path}")
    
    return sequences, targets, relevance_scores, stock_indices

def create_dataset(data, features, sequence_length, ranking_data_path=None):
    """保持原有接口，但内部调用新的排序数据集创建函数"""
    return create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path)

def create_ranking_dataset_vectorized(data, features, sequence_length, ranking_data_path=None, min_window_end_date=None, market_normalizer=None):
    """
    向量化加速版本：预计算每只股票的所有滑动窗口，再按日期聚合。
    保持与原函数完全相同的输出格式。
    
    Args:
        market_normalizer: MarketNormalizer实例，用于分层标准化收益率。若为None则使用原始排名逻辑。
    """
    # if ranking_data_path and os.path.exists(ranking_data_path):
    #     print(f"加载已有的排序数据集: {ranking_data_path}")
    #     return joblib.load(ranking_data_path)

    print("正在创建排序数据集（向量化加速版本）...")
    # data.rename(columns={'stock_idx': 'instrument'}, inplace=True)
    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])

    # 1. 确保数据按股票和时间排序
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    
    # 2. 确保每只股票都有 'label'（次日涨跌幅），否则无法作为 target
    data = data.dropna(subset=['label'])
    
    # 3. 为每只股票生成所有滑动窗口
    # 仅保留满足以下条件的 end_date：
    # - 历史窗口长度满足 sequence_length
    # - end_date 之后存在 5 条未来数据
    # - 这 5 条未来数据在自然日上连续（任意节假日/周末导致的日期跳跃都会被过滤）
    all_windows = []  # 每个元素: (end_date, stock_code, sequence, target)

    print("Step 1: 为每只股票生成滑动窗口...")
    grouped = data.groupby('instrument')
    
    for stock_code, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length:
            continue
        
        # 提取特征和 label
        feature_values = group[features].values.astype(np.float32)  # (T, F)
        labels = group['label'].values.astype(np.float32)           # (T,)
        dates = group['datetime'].values                            # (T,)
        dates_day = group['datetime'].values.astype('datetime64[D]')

        # 生成滑动窗口：从第 sequence_length-1 行开始（0-indexed）
        num_windows = len(group) - sequence_length + 1
        n = len(group)
        for i in range(num_windows):
            end_idx = i + sequence_length - 1

            # 需要有未来 5 条数据
            if end_idx + 5 >= n:
                continue

            # 未来5条数据间最大自然日间隔：≤3天允许正常周末，>3天则可能是长假/停牌
            future_dates = dates_day[end_idx + 1:end_idx + 6]
            future_diffs = np.diff(future_dates).astype(np.int64)
            if not (np.all(future_diffs > 0) and np.all(future_diffs <= 3)):
                continue

            seq = feature_values[i : i + sequence_length]   # (L, F)
            target = labels[end_idx]                        # label 对应窗口最后一天的次日涨跌幅
            end_date = dates[end_idx]                       # 窗口结束日期（即预测日）
            all_windows.append((end_date, stock_code, seq, target))

    # 4. 转为 DataFrame 便于按日期聚合
    print("Step 2: 按日期聚合窗口...")
    window_df = pd.DataFrame(all_windows, columns=['date', 'stock_code', 'seq', 'target'])

    # 5. 按 date 分组，构建每日样本
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []
    sample_dates = []

    print("Step 3: 构建每日样本并计算 relevance...")
    grouped_by_date = window_df.groupby('date')

    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)
    
    for date, group in tqdm(grouped_by_date, desc="Aggregating by date"):
        if min_window_end_date is not None and pd.to_datetime(date) < min_window_end_date:
            continue

        if len(group) < 10:
            continue
        
        # 当日股票数过多时随机采样，防止单日array过大导致内存溢出
        max_stocks = 200
        if len(group) > max_stocks:
            group = group.sample(max_stocks, random_state=42)
        
        # 提取数据
        day_seqs = np.stack(group['seq'].values)          # (N, L, F)
        day_targets = group['target'].values              # (N,)
        day_stocks = group['stock_code'].tolist()         # [str]

        # 计算 relevance（分层标准化收益率或原始排名）
        if market_normalizer is not None:
            # 创建包含日期和target的DataFrame用于标准化
            day_df = pd.DataFrame({
                'datetime': [date] * len(day_targets),
                'label': day_targets
            })
            relevance = market_normalizer.transform(day_df).astype(np.float32)
        else:
            # 回退到原始排名逻辑
            sorted_indices = np.argsort(day_targets)[::-1]
            relevance = np.zeros_like(day_targets, dtype=np.float32)
            for rank, idx in enumerate(sorted_indices):
                relevance[idx] = len(day_targets) - rank

        sequences.append(day_seqs)
        targets.append(day_targets)
        relevance_scores.append(relevance)
        stock_indices.append(day_stocks)
        sample_dates.append(str(date))

    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        avg_stocks = np.mean([len(seq) for seq in sequences])
        print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")

    # 6. 保存
    # if ranking_data_path:
    #     joblib.dump((sequences, targets, relevance_scores, stock_indices), ranking_data_path)
    #     print(f"数据集已保存到: {ranking_data_path}")

    return sequences, targets, relevance_scores, stock_indices, sample_dates


def create_ranking_dataset_streaming(data, features, sequence_length, output_dir,
                                      min_window_end_date=None, market_normalizer=None,
                                      future_max_gap=5):
    """
    流式内存优化版：仅存储窗口元数据，逐日提取序列并写入 .npz。
    
    设计要点：
      - 窗口构建阶段仅存 (date, stock_code, start_idx, end_idx, target) 元数据
      - 股票特征数组保留在 stock_feature_cache 中供按需切片
      - 逐日 stack → 计算 relevance → 写入 .npz，不在内存中累积全量序列
      - future_max_gap 控制 T+1~T+5 间允许的最大自然日间隔（默认5天，含周末+3天假期）
    
    Returns:
        npz_paths:   list of .npz file paths, one per date
        dates_list:  list of date strings
        stock_idx_list: list of lists of instrument indices (for prior_bias computation)
    """
    os.makedirs(output_dir, exist_ok=True)
    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    data = data.dropna(subset=['label'])

    sequence_length = int(sequence_length)
    all_windows = []  # (end_date, stock_code, start_idx, end_idx, target)
    stock_feature_cache = {}  # stock_code -> (feature_values, labels)

    print("Step 1: 为每只股票生成滑动窗口（仅元数据）...")
    grouped = data.groupby('instrument')

    for stock_code, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length:
            continue

        feature_values = group[features].values.astype(np.float32)  # (T, F)
        labels = group['label'].values.astype(np.float32)            # (T,)
        dates = group['datetime'].values
        dates_day = dates.astype('datetime64[D]')
        n = len(group)
        num_windows = n - sequence_length + 1

        stock_feature_cache[stock_code] = (feature_values, labels)

        for i in range(num_windows):
            end_idx = i + sequence_length - 1
            if end_idx + 5 >= n:
                continue

            # T+1~T+5 间最大自然日间隔，允许周末(3)+3天假期(4~5)，排除长假(≥6)
            future_dates = dates_day[end_idx + 1:end_idx + 6]
            future_diffs = np.diff(future_dates).astype(np.int64)
            if not (np.all(future_diffs > 0) and np.all(future_diffs <= future_max_gap)):
                continue

            target = labels[end_idx]
            end_date = dates[end_idx]
            all_windows.append((end_date, stock_code, i, end_idx, target))

    print(f"  共发现 {len(all_windows)} 个有效窗口（元数据 {len(all_windows) * 40 / 1024 / 1024:.1f} MB）")

    # 按日期分组
    print("Step 2: 按日期聚合...")
    window_df = pd.DataFrame(all_windows, columns=['date', 'stock_code', 'start_idx', 'end_idx', 'target'])
    del all_windows  # 释放元数据列表内存
    grouped_by_date = window_df.groupby('date')

    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)

    npz_paths = []
    dates_list = []
    stock_idx_list = []

    pbar = tqdm(grouped_by_date, desc="逐日生成 .npz")
    for date, group in pbar:
        if min_window_end_date is not None and pd.to_datetime(date) < min_window_end_date:
            continue
        if len(group) < 10:
            continue

        stocks = group['stock_code'].values.astype(np.int64)
        si_arr = group['start_idx'].values.astype(np.int64)
        ei_arr = group['end_idx'].values.astype(np.int64)
        day_targets = group['target'].values.astype(np.float32)
        day_seqs = [stock_feature_cache[s][0][si_arr[j]:ei_arr[j] + 1]
                    for j, s in enumerate(stocks)]

        day_seqs = np.stack(day_seqs).astype(np.float32)

        # 计算 relevance
        if market_normalizer is not None:
            day_df = pd.DataFrame({'datetime': [date] * len(day_targets), 'label': day_targets})
            relevance = market_normalizer.transform(day_df).astype(np.float32)
        else:
            sorted_indices = np.argsort(day_targets)[::-1]
            relevance = np.zeros_like(day_targets, dtype=np.float32)
            for rank, idx in enumerate(sorted_indices):
                relevance[idx] = len(day_targets) - rank

        # 写入 .npz（非压缩模式，I/O 快 3-4x）
        date_str = str(date)[:10]
        path = os.path.join(output_dir, f"{date_str}.npz")
        np.savez(path, sequences=day_seqs, targets=day_targets,
                 relevance=relevance, stock_indices=stocks)
        npz_paths.append(path)
        dates_list.append(date_str)
        stock_idx_list.append(stocks)

        pbar.set_postfix_str(f"{date_str} N={len(stocks)}")

    print(f"成功创建 {len(npz_paths)} 个日样本 (.npz 文件)")
    if len(npz_paths) > 0:
        avg_stocks = np.mean([len(s) for s in stock_idx_list])
        print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")

    return npz_paths, dates_list, stock_idx_list


def create_ranking_dataset_inmemory(data, features, sequence_length,
                                     market_normalizer=None, future_max_gap=5,
                                     macro_lookup=None):
    """
    全内存版：仅保留窗口元数据和股票特征缓存，不写磁盘。

    Args:
        macro_lookup: 可选 dict {date_str: (market_state[2], industry_rotation[K], stock_beta[N])}.
            每日 per-day macro 特征, 供 MD-SRP gate 使用.

    Returns:
        dict with keys:
            window_data:    list of dicts, each with stock_indices/start_indices/end_indices/targets/relevance/macro
            feature_cache:  dict[stock_idx -> (features_array, labels_array)]
            seq_len:        int, sequence length
            dates:          list of date strings
            stock_indices:  list of np.array (per-date stock indices)
    """
    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    data = data.dropna(subset=['label'])

    sequence_length = int(sequence_length)
    all_windows = []  # (end_date, stock_code, start_idx, end_idx, target)
    stock_feature_cache = {}

    print("Step 1: 为每只股票生成滑动窗口（仅元数据）...")
    grouped = data.groupby('instrument')
    for stock_code, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length:
            continue
        feature_values = group[features].values.astype(np.float32)
        labels = group['label'].values.astype(np.float32)
        dates = group['datetime'].values
        dates_day = dates.astype('datetime64[D]')
        n = len(group)
        num_windows = n - sequence_length + 1

        stock_feature_cache[stock_code] = (feature_values, labels)

        for i in range(num_windows):
            end_idx = i + sequence_length - 1
            if end_idx + 5 >= n:
                continue
            future_dates = dates_day[end_idx + 1:end_idx + 6]
            future_diffs = np.diff(future_dates).astype(np.int64)
            if not (np.all(future_diffs > 0) and np.all(future_diffs <= future_max_gap)):
                continue
            target = labels[end_idx]
            end_date = dates[end_idx]
            all_windows.append((end_date, stock_code, i, end_idx, target))

    print(f"  共发现 {len(all_windows)} 个有效窗口")
    print(f"  feature_cache 内存: {sum(v[0].nbytes + v[1].nbytes for v in stock_feature_cache.values()) / 1024 / 1024:.1f} MB")

    print("Step 2: 按日期聚合...")
    window_df = pd.DataFrame(all_windows, columns=['date', 'stock_code', 'start_idx', 'end_idx', 'target'])
    del all_windows
    grouped_by_date = window_df.groupby('date')

    window_data = []
    dates_list = []
    stock_idx_list = []

    pbar = tqdm(sorted(grouped_by_date), desc="计算 relevance + 组装元数据")
    for date, group in pbar:
        if len(group) < 10:
            continue

        stocks = group['stock_code'].values.astype(np.int64)
        si_arr = group['start_idx'].values.astype(np.int64)
        ei_arr = group['end_idx'].values.astype(np.int64)
        targets = group['target'].values.astype(np.float32)

        if market_normalizer is not None:
            day_df = pd.DataFrame({'datetime': [date] * len(targets), 'label': targets})
            relevance = market_normalizer.transform(day_df).astype(np.float32)
        else:
            sorted_indices = np.argsort(targets)[::-1]
            relevance = np.zeros(len(targets), dtype=np.float32)
            for rank, idx in enumerate(sorted_indices):
                relevance[idx] = len(targets) - rank

        # macro 特征 (per day)
        date_str = str(date)[:10]
        if macro_lookup is not None and date_str in macro_lookup:
            ms, ir, sb = macro_lookup[date_str]
        else:
            ms = np.zeros(2, dtype=np.float32)
            ir = np.zeros(11, dtype=np.float32)
            sb = np.zeros(len(stocks), dtype=np.float32)

        window_meta = {
            'stock_indices': stocks,
            'start_indices': si_arr,
            'end_indices': ei_arr,
            'targets': targets,
            'relevance': relevance,
            'market_state': ms.astype(np.float32),
            'industry_rotation': ir.astype(np.float32),
            'stock_beta': sb.astype(np.float32),
        }
        window_data.append(window_meta)
        dates_list.append(date_str)
        stock_idx_list.append(stocks)
        pbar.set_postfix_str(f"{date_str} N={len(stocks)}")

    print(f"  成功构建 {len(window_data)} 个窗口元数据")
    return {
        'window_data': window_data,
        'feature_cache': stock_feature_cache,
        'seq_len': sequence_length,
        'dates': dates_list,
        'stock_indices': stock_idx_list,
    }


class InMemoryRankingDataset:
    """内存版 RankingDataset: 从 window_data 元数据 + feature_cache 缓存按需构建序列.

    Returns (per item):
        sequences:      (n_stocks, seq_len, n_features) float32
        targets:        (n_stocks,) float32, 当日真实 5d 收益
        relevance:      (n_stocks,) float32, market_normalizer 标准化后的排序目标
        stock_indices:  (n_stocks,) int64
    collate_fn 负责 padding 到 batch 内 max_stocks 并生成 masks.
    """
    def __init__(self, window_data, feature_cache, seq_len):
        self.window_data = window_data
        self.feature_cache = feature_cache
        self.seq_len = int(seq_len)
        self.feature_dim = None
        for stock_code, (feats, _) in feature_cache.items():
            self.feature_dim = int(feats.shape[1])
            break
        if self.feature_dim is None:
            raise ValueError("feature_cache is empty")
        # 预存 macro 特征 (per window)
        self.market_states = [m['market_state'] for m in window_data]
        self.industry_rots = [m['industry_rotation'] for m in window_data]
        self.stock_betas = [m['stock_beta'] for m in window_data]

    def __len__(self):
        return len(self.window_data)

    def __getitem__(self, idx):
        import torch
        meta = self.window_data[idx]
        stocks = meta['stock_indices']
        si = meta['start_indices']
        ei = meta['end_indices']
        targets = meta['targets']
        relevance = meta['relevance']

        n = len(stocks)
        seqs = np.zeros((n, self.seq_len, self.feature_dim), dtype=np.float32)
        for i, stock_code in enumerate(stocks):
            feats, _ = self.feature_cache[int(stock_code)]
            seqs[i] = feats[si[i]:ei[i] + 1]

        return {
            'sequences': torch.from_numpy(seqs),
            'targets': torch.from_numpy(targets.astype(np.float32)),
            'relevance': torch.from_numpy(relevance.astype(np.float32)),
            'stock_indices': torch.from_numpy(stocks.astype(np.int64)),
            'market_state': torch.from_numpy(self.market_states[idx]),
            'industry_rotation': torch.from_numpy(self.industry_rots[idx]),
            'stock_beta': torch.from_numpy(self.stock_betas[idx]),
        }