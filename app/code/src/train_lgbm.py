import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import config
from featurework import engineer_features_158plus39

try:
    import lightgbm as lgb
except ImportError:
    print("请安装lightgbm: pip install lightgbm")
    sys.exit(1)

LGBM_CONFIG = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
    'early_stopping_rounds': 50,
    'eval_metric': 'ndcg',
}

FEATURE_COLUMNS = [
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
]

def _build_label_lgbm(processed, drop_small_open=True):
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]
    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed

def preprocess_lgbm(df, stockid2idx, is_train=True):
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    
    import multiprocessing as mp
    from tqdm import tqdm
    
    groups = []
    for stock_code, group in df.groupby('股票代码', sort=False):
        if len(group) >= config['sequence_length']:
            groups.append(group)
    
    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(engineer_features_158plus39, groups), total=len(groups), desc="特征工程"))
    
    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    
    drop_small_open = is_train
    processed = _build_label_lgbm(processed, drop_small_open=drop_small_open)
    
    return processed

def create_flat_dataset(data, features, sequence_length):
    print("创建扁平化数据集...")
    data = data.copy()
    data = data.sort_values(['instrument', '日期']).reset_index(drop=True)
    data = data.dropna(subset=['label'])
    
    all_samples = []
    
    grouped = data.groupby('instrument')
    for stock_code, group in tqdm(grouped, desc="处理股票"):
        if len(group) < sequence_length:
            continue
        
        feature_values = group[features].values.astype(np.float32)
        labels = group['label'].values.astype(np.float32)
        dates = group['日期'].values
        dates_day = group['日期'].values.astype('datetime64[D]')
        
        num_windows = len(group) - sequence_length + 1
        n = len(group)
        for i in range(num_windows):
            end_idx = i + sequence_length - 1
            if end_idx + 5 >= n:
                continue
            future_dates = dates_day[end_idx + 1:end_idx + 6]
            future_diffs = np.diff(future_dates).astype(np.int64)
            if not np.all((future_diffs > 0) & (future_diffs <= 7)):
                continue
            features_last_day = feature_values[end_idx]
            label = labels[end_idx]
            date = dates[end_idx]
            all_samples.append((date, stock_code, features_last_day, label))
    
    if len(all_samples) == 0:
        raise ValueError("没有可用的样本")
    
    samples_df = pd.DataFrame(all_samples, columns=['date', 'stock_id', 'features', 'label'])
    samples_df = samples_df.sort_values('date').reset_index(drop=True)
    
    samples_df['rank_label'] = 0
    for date, group in samples_df.groupby('date'):
        sorted_indices = np.argsort(group['label'].values)[::-1]
        ranks = np.arange(1, len(sorted_indices) + 1)
        samples_df.loc[group.index, 'rank_label'] = ranks[np.argsort(sorted_indices)]
    
    X = np.stack(samples_df['features'].values)
    y = samples_df['rank_label'].values.astype(int)
    dates = samples_df['date'].values
    stock_ids = samples_df['stock_id'].values
    
    date_counts = samples_df.groupby('date').size()
    groups_list = date_counts.values.tolist()
    
    print(f"特征维度: {X.shape}")
    print(f"日期范围: {dates.min()} 到 {dates.max()}")
    print(f"分组数量: {len(groups_list)}")
    
    return X, y, groups_list, dates, stock_ids

def train_lgbm():
    print("=" * 60)
    print("LGBM排序模型训练")
    print("=" * 60)
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = config['data_path']
    train_path = os.path.join(data_path, 'train.csv')
    
    print(f"加载训练数据: {train_path}")
    train_df = pd.read_csv(train_path, dtype={"股票代码": str})
    train_df["股票代码"] = train_df["股票代码"].astype(str).str.zfill(6)
    train_df["日期"] = pd.to_datetime(train_df["日期"])
    
    all_stock_ids = sorted(train_df["股票代码"].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}
    
    print("预处理数据...")
    train_processed = preprocess_lgbm(train_df, stockid2idx, is_train=True)
    train_processed = train_processed.rename(columns={'日期': 'datetime'})
    
    all_dates = pd.to_datetime(train_processed['datetime'])
    last_date = all_dates.max()
    val_start = last_date - pd.DateOffset(months=1)
    
    print(f"验证集开始日期: {val_start.date()}")
    
    X, y, groups, dates, stock_ids = create_flat_dataset(
        train_processed,
        FEATURE_COLUMNS,
        config['sequence_length']
    )
    
    train_mask = pd.to_datetime(dates) < val_start
    val_mask = pd.to_datetime(dates) >= val_start
    
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    train_dates, val_dates = dates[train_mask], dates[val_mask]
    
    train_date_counts = pd.Series(train_dates).value_counts().sort_index()
    val_date_counts = pd.Series(val_dates).value_counts().sort_index()
    train_groups = train_date_counts.values.tolist()
    val_groups = val_date_counts.values.tolist()
    
    print(f"训练集样本数: {len(X_train)}, 日期数: {len(train_groups)}")
    print(f"验证集样本数: {len(X_val)}, 日期数: {len(val_groups)}")
    
    train_data = lgb.Dataset(X_train, label=y_train, group=train_groups, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_data, free_raw_data=False)
    
    lgbm_params = {
        'objective': 'lambdarank',
        'metric': LGBM_CONFIG['eval_metric'],
        'ndcg_eval_at': [5],
        'boosting_type': 'gbdt',
        'num_leaves': LGBM_CONFIG['num_leaves'],
        'max_depth': LGBM_CONFIG['max_depth'],
        'learning_rate': LGBM_CONFIG['learning_rate'],
        'n_estimators': LGBM_CONFIG['n_estimators'],
        'subsample': LGBM_CONFIG['subsample'],
        'colsample_bytree': LGBM_CONFIG['colsample_bytree'],
        'reg_alpha': LGBM_CONFIG['reg_alpha'],
        'reg_lambda': LGBM_CONFIG['reg_lambda'],
        'min_child_samples': LGBM_CONFIG['min_child_samples'],
        'min_child_weight': LGBM_CONFIG['min_child_weight'],
        'random_state': LGBM_CONFIG['random_state'],
        'n_jobs': LGBM_CONFIG['n_jobs'],
        'verbosity': LGBM_CONFIG['verbosity'],
        'label_gain': list(range(300)),
    }
    
    print("开始训练...")
    model = lgb.train(
        lgbm_params,
        train_data,
        valid_sets=[val_data],
        valid_names=['valid'],
        num_boost_round=LGBM_CONFIG['n_estimators'],
        callbacks=[
            lgb.early_stopping(LGBM_CONFIG['early_stopping_rounds']),
            lgb.log_evaluation(50)
        ]
    )
    
    model_path = os.path.join(output_dir, 'lgbm_model.pkl')
    joblib.dump(model, model_path)
    print(f"模型已保存: {model_path}")
    
    print("\n训练完成!")
    return model

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    train_lgbm()