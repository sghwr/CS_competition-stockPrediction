"""
LGBM数据加载器
复用现有特征工程和预处理，修正标签计算为 T+1 到 T+5 收益率
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import multiprocessing as mp

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "code", "src"))

# 导入现有模块
from config import config as base_config
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_vectorized
from train import _preprocess_common, preprocess_data, preprocess_val_data

# 特征工程映射
feature_cloums_map = {
    '39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv','volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],
    '158+39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}

feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}


def _build_label_lgbm(processed, drop_small_open=True):
    """
    LGBM专用标签计算：从T+1开盘到T+5开盘的收益率
    与竞赛实际交易场景一致
    """
    # 使用T+1日开盘价作为买入价，T+5日开盘价作为卖出价
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)  # T+1开盘价
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)  # T+5开盘价
    
    # 过滤无效开盘价
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]
    
    # 计算收益率
    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])
    
    # 清理临时列
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def preprocess_data_lgbm(df, stockid2idx, config, is_train=True):
    """
    LGBM专用数据预处理
    复用现有特征工程，但使用修正后的标签计算
    """
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]
    
    # 保证时序正确
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    
    desc = "训练集特征工程" if is_train else "特征工程"
    print(f"正在使用多进程进行{desc}...")
    
    # 过滤掉数据点不足的股票
    groups = []
    dropped_stocks = []
    for stock_code, group in df.groupby('股票代码', sort=False):
        if len(group) >= config['sequence_length']:
            groups.append(group)
        else:
            dropped_stocks.append(stock_code)
    
    if dropped_stocks:
        print(f"  过滤掉 {len(dropped_stocks)} 只数据不足的股票: {dropped_stocks[:10]}...")
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")
    
    # 多进程特征工程
    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))
    
    processed = pd.concat(processed_list).reset_index(drop=True)
    
    # 映射股票索引
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    
    # 使用LGBM专用标签计算
    drop_small_open = is_train  # 训练集过滤小开盘价，验证集不过滤
    processed = _build_label_lgbm(processed, drop_small_open=drop_small_open)
    
    # 动态提取特征列（按照预定义顺序，只保留实际存在的特征）
    predefined_features = feature_cloums_map[config['feature_num']]
    # 非特征列
    non_feature_cols = ['instrument', '股票代码', '日期', 'label']
    # 只保留实际存在且不是非特征列的预定义特征
    dynamic_feature_columns = [
        col for col in predefined_features 
        if col in processed.columns and col not in non_feature_cols
    ]
    
    # 检查是否有特征缺失
    missing_features = set(predefined_features) - set(dynamic_feature_columns) - set(non_feature_cols)
    if missing_features:
        print(f"警告: 缺失 {len(missing_features)} 个预定义特征: {list(missing_features)[:10]}...")
    
    # 确保特征列不为空
    if len(dynamic_feature_columns) == 0:
        raise ValueError("未找到有效的特征列")
    
    print(f"动态提取特征列: {len(dynamic_feature_columns)} 个特征")
    
    return processed, dynamic_feature_columns


def create_flat_dataset_for_lgbm(data, features, sequence_length, market_normalizer=None):
    """
    为LGBM创建扁平化数据集
    每个(股票,日期)为独立样本，返回特征矩阵和标签向量
    同时返回分组信息（每个日期的股票数量），用于LambdaRank
    
    Args:
        data: 预处理后的DataFrame，包含'instrument', 'datetime', 'label'和特征列
        features: 特征列名列表
        sequence_length: 序列长度
        market_normalizer: MarketNormalizer实例，用于标准化标签（可选）
    
    Returns:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签向量 (n_samples,)
        groups: 分组向量，每个元素表示该组的样本数
        dates: 日期向量 (n_samples,)
        stock_ids: 股票ID向量 (n_samples,)
    """
    print("正在为LGBM创建扁平化数据集...")
    
    # 确保数据按股票和时间排序
    data = data.copy()
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    
    # 为每只股票生成滑动窗口
    all_samples = []  # 每个元素: (date, stock_id, features, label)
    
    grouped = data.groupby('instrument')
    
    for stock_id, group in tqdm(grouped, desc="处理股票"):
        if len(group) < sequence_length:
            continue
        
        # 提取特征和标签
        feature_values = group[features].values.astype(np.float32)  # (T, F)
        labels = group['label'].values.astype(np.float32)           # (T,)
        dates = group['datetime'].values                            # (T,)
        dates_day = group['datetime'].values.astype('datetime64[D]')
        
        # 生成滑动窗口
        num_windows = len(group) - sequence_length + 1
        n = len(group)
        
        for i in range(num_windows):
            end_idx = i + sequence_length - 1
            
            # 需要有未来5条数据（标签已计算，这里只需检查边界）
            if end_idx >= n:
                continue
            
            # 使用窗口最后一天的特征（即预测日的特征）
            # 对于LGBM，我们使用最后一天的特征，而不是整个序列
            features_last_day = feature_values[end_idx]  # (F,)
            label = labels[end_idx]                      # 标量
            date = dates[end_idx]                        # 日期
            
            # 检查未来5日连续性（与原始逻辑一致）
            if end_idx + 5 >= n:
                continue
            
            future_dates = dates_day[end_idx + 1:end_idx + 6]
            future_diffs = np.diff(future_dates).astype(np.int64)
            if not np.all((future_diffs > 0) & (future_diffs <= 7)):
                continue
            
            all_samples.append((date, stock_id, features_last_day, label))
    
    if len(all_samples) == 0:
        raise ValueError("没有可用的样本")
    
    print(f"共生成 {len(all_samples)} 个样本")
    
    # 转换为DataFrame便于按日期分组
    samples_df = pd.DataFrame(all_samples, columns=['date', 'stock_id', 'features', 'label'])
    
    # 按日期排序
    samples_df = samples_df.sort_values('date').reset_index(drop=True)
    
    # 将浮点数标签转换为整数排名（LambdaRank要求整数标签）
    # 对每个日期，按收益率从高到低排序，分配排名分数（1到N）
    print("将浮点数标签转换为整数排名...")
    samples_df['rank_label'] = 0
    
    for date, group in samples_df.groupby('date'):
        # 按标签降序排序（收益率越高排名越高）
        sorted_indices = np.argsort(group['label'].values)[::-1]  # 降序
        ranks = np.arange(1, len(sorted_indices) + 1)
        # 分配排名
        samples_df.loc[group.index, 'rank_label'] = ranks[np.argsort(sorted_indices)]
    
    # 提取数据
    X = np.stack(samples_df['features'].values)  # (n_samples, n_features)
    y = samples_df['rank_label'].values.astype(int)  # 整数排名标签
    dates = samples_df['date'].values            # (n_samples,)
    stock_ids = samples_df['stock_id'].values    # (n_samples,)
    
    # 计算分组信息（每个日期的样本数）
    date_counts = samples_df.groupby('date').size()
    groups = date_counts.values.tolist()  # 每个日期的样本数列表
    
    print(f"特征维度: {X.shape}")
    print(f"日期范围: {dates.min()} 到 {dates.max()}")
    print(f"分组数量: {len(groups)}")
    print(f"平均每组样本数: {np.mean(groups):.1f}")
    print(f"标签范围: {y.min()} 到 {y.max()}")
    
    return X, y, groups, dates, stock_ids


def load_and_prepare_data(config):
    """
    加载并准备LGBM训练数据
    """
    data_path = config['data_path']
    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')
    
    print(f"加载训练数据: {train_path}")
    train_df = pd.read_csv(train_path, dtype={"股票代码": str})
    train_df["股票代码"] = train_df["股票代码"].astype(str).str.zfill(6)
    train_df["日期"] = pd.to_datetime(train_df["日期"])
    
    print(f"加载测试数据: {test_path}")
    test_df = pd.read_csv(test_path, dtype={"股票代码": str})
    test_df["股票代码"] = test_df["股票代码"].astype(str).str.zfill(6)
    test_df["日期"] = pd.to_datetime(test_df["日期"])
    
    # 获取所有股票ID映射
    all_stock_ids = sorted(train_df["股票代码"].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}
    num_stocks = len(stockid2idx)
    print(f"股票总数: {num_stocks}")
    
    # 预处理训练数据
    train_processed, feature_columns = preprocess_data_lgbm(
        train_df, stockid2idx, config, is_train=True
    )
    
    # 重命名日期列以匹配create_ranking_dataset_vectorized的期望
    train_processed = train_processed.rename(columns={'日期': 'datetime'})
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'train_processed': train_processed,
        'feature_columns': feature_columns,
        'stockid2idx': stockid2idx,
        'num_stocks': num_stocks,
    }


if __name__ == "__main__":
    # 测试数据加载
    print("数据加载器测试...")
    # 使用基本配置进行测试
    test_config = {
        'sequence_length': 60,
        'feature_num': '158+39',
        'data_path': '../data',
    }
    
    try:
        data_dict = load_and_prepare_data(test_config)
        print(f"特征列数: {len(data_dict['feature_columns'])}")
        print(f"训练数据样本数: {len(data_dict['train_processed'])}")
        print("数据加载测试通过!")
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()