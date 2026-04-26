import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm

def _rolling_linear_regression(x, y):
    x = np.vstack([np.ones(len(x)), x]).T
    beta, res, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta[1], res[0] if len(res) > 0 else 0.0, np.sum((y - (x @ beta))**2)

def engineer_features_158plus39(df):
    df_copy = df.copy()
    df_158 = engineer_features(df_copy)
    df_39 = engineer_features_39(df_copy)
    
    feature_cols_39 = [
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 
        'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 
        'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 
        'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ]
    
    feature_cols_39_exist = [col for col in feature_cols_39 if col in df_39.columns]
    df_final = pd.concat([df_158, df_39[feature_cols_39_exist]], axis=1)
    df_final = df_final.loc[:,~df_final.columns.duplicated()]
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)
    
    return df_final

def engineer_features_39(df):
    try:
        import talib
        import numpy as np
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    df = df.copy()

    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    df['sma_5'] = talib.SMA(close, timeperiod=5)
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['ema_26'] = talib.EMA(close, timeperiod=26)
    df['ema_60'] = talib.EMA(close, timeperiod=60)

    macd_line, macd_signal_line, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal_line

    df['rsi'] = talib.RSI(close, timeperiod=14)

    df['kdj_k'], df['kdj_d'] = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    df['boll_mid'], df['boll_upper'], df['boll_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['boll_std'] = (df['boll_upper'] - df['boll_mid']) / 2
    df.drop(columns=['boll_upper', 'boll_lower'], inplace=True)

    df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

    df['obv'] = talib.OBV(close, volume)

    df['volume_change'] = volume.pct_change()
    df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
    df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = df['volume_ma_5'] / df['volume_ma_20']

    df['return_1'] = close.pct_change(1)
    df['return_5'] = close.pct_change(5)
    df['return_10'] = close.pct_change(10)
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()

    df['high_low_spread'] = high - low
    df['open_close_spread'] = open_ - close
    df['high_close_spread'] = high - close
    df['low_close_spread'] = low - close

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

def engineer_features(df):
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    df = df.copy()

    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    vwap = df['成交额'] / (volume + 1e-12)

    features = []
    feature_names = []

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

    features.extend([
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12)
    ])
    feature_names.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])

    windows = [5, 10, 20, 30, 60]

    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f'ROC{w}')

    for w in windows:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MA{w}')

    for w in windows:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'STD{w}')

    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f'BETA{w}')
        
        n = min(w, len(close))
        time_period_series = pd.Series(range(n), index=close.index[:n])
        rolling_corr = close.rolling(w).corr(time_period_series)
        rsquare = rolling_corr**2
        features.append(rsquare)
        feature_names.append(f'RSQR{w}')

        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
        predicted = slope * (w - 1) + intercept
        resi = close - predicted
        features.append(resi / (close + 1e-12))
        feature_names.append(f'RESI{w}')

    for w in windows:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MAX{w}')
    for w in windows:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MIN{w}')

    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f'QTLU{w}')
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f'QTLD{w}')

    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f'RANK{w}')

    for w in windows:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        feature_names.append(f'RSV{w}')

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

    log_volume = np.log(volume + 1)
    for w in windows:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        feature_names.append(f'CORR{w}')
    
    close_ret = close / close.shift(1)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    log_volume_ret = np.log(volume_ret + 1)
    for w in windows:
        corr_df = pd.concat([close_ret, log_volume_ret], axis=1).fillna(0)
        features.append(talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w))
        feature_names.append(f'CORD{w}')

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

    for w in windows:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VMA{w}')
    for w in windows:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VSTD{w}')

    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in windows:
        mean_vol_w_ret = vol_weighted_ret.rolling(w).mean()
        std_vol_w_ret = vol_weighted_ret.rolling(w).std()
        features.append(std_vol_w_ret / (mean_vol_w_ret + 1e-12))
        feature_names.append(f'WVMA{w}')

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

    feature_df = pd.concat(features, axis=1)
    feature_df.columns = feature_names
    
    df = pd.concat([df, feature_df], axis=1)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def create_ranking_dataset_vectorized(data, features, sequence_length, ranking_data_path=None, min_window_end_date=None, market_normalizer=None):
    print("正在创建排序数据集（向量化加速版本）...")
    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])

    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    data = data.dropna(subset=['label'])
    
    all_windows = []

    print("Step 1: 为每只股票生成滑动窗口...")
    grouped = data.groupby('instrument')
    
    for stock_code, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length:
            continue
        
        feature_values = group[features].values.astype(np.float32)
        labels = group['label'].values.astype(np.float32)
        dates = group['datetime'].values
        dates_day = group['datetime'].values.astype('datetime64[D]')

        num_windows = len(group) - sequence_length + 1
        n = len(group)
        for i in range(num_windows):
            end_idx = i + sequence_length - 1

            if end_idx + 5 >= n:
                continue

            future_dates = dates_day[end_idx + 1:end_idx + 6]
            future_diffs = np.diff(future_dates).astype(np.int64)
            if not np.all(future_diffs == 1):
                continue

            seq = feature_values[i : i + sequence_length]
            target = labels[end_idx]
            end_date = dates[end_idx]
            all_windows.append((end_date, stock_code, seq, target))

    print("Step 2: 按日期聚合窗口...")
    window_df = pd.DataFrame(all_windows, columns=['date', 'stock_code', 'seq', 'target'])

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
        
        day_seqs = np.stack(group['seq'].values)
        day_targets = group['target'].values
        day_stocks = group['stock_code'].tolist()

        if market_normalizer is not None:
            day_df = pd.DataFrame({
                'datetime': [date] * len(day_targets),
                'label': day_targets
            })
            relevance = market_normalizer.transform(day_df).astype(np.float32)
        else:
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

    return sequences, targets, relevance_scores, stock_indices, sample_dates