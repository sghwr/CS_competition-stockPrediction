"""
统一数据加载器
为Transformer和LGBM准备不同格式的数据
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm

# 添加必要的路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'LGBM'))

# 导入现有模块
try:
    from config import config as transformer_config
    from utils import engineer_features_39, engineer_features_158plus39
    from LGBM.data_loader import load_and_prepare_data as load_lgbm_data
    from LGBM.data_loader import preprocess_data_lgbm
    from code.src.predict import preprocess_predict_data, build_inference_sequences
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保code/src和LGBM目录存在且可访问")
    raise

from .config import ENSEMBLE_CONFIG, FEATURE_COLUMNS_MAP, FEATURE_ENGINEER_MAP


class EnsembleDataLoader:
    """为集成模型准备数据的统一加载器"""
    
    def __init__(self, config=None):
        self.config = config or ENSEMBLE_CONFIG
        self.feature_num = self.config['feature_num']
        self.sequence_length = self.config['sequence_length']
        
    def load_raw_data(self):
        """加载原始数据"""
        train_path = self.config['data_paths']['train']
        test_path = self.config['data_paths']['test']
        
        if not train_path.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {train_path}")
        
        train_df = pd.read_csv(train_path, dtype={'股票代码': str})
        
        # 如果有测试数据也加载
        test_df = None
        if test_path.exists():
            test_df = pd.read_csv(test_path, dtype={'股票代码': str})
        
        # 统一股票代码格式
        for df in [train_df, test_df]:
            if df is not None:
                df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
                df['日期'] = pd.to_datetime(df['日期'])
        
        return train_df, test_df
    
    def prepare_for_transformer(self, df, scaler=None):
        """为Transformer准备4D序列数据"""
        # 获取股票ID映射
        stock_ids = sorted(df['股票代码'].unique())
        stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
        
        # 特征工程
        processed, features = preprocess_predict_data(df, stockid2idx)
        
        # 标准化
        if scaler:
            processed[features] = scaler.transform(processed[features])
        
        # 构建序列
        latest_date = processed['日期'].max()
        sequences, sequence_stock_ids = build_inference_sequences(
            processed, features, self.sequence_length, stock_ids, latest_date
        )
        
        # 转换为4D张量 [1, n_stocks, seq_len, n_features]
        sequences_tensor = torch.from_numpy(sequences).unsqueeze(0).float()
        
        return {
            'sequences': sequences_tensor,
            'stock_ids': sequence_stock_ids,
            'features': features,
            'latest_date': latest_date
        }
    
    def prepare_for_lgbm(self, df):
        """为LGBM准备2D扁平数据"""
        # 获取股票ID映射
        stock_ids = sorted(df['股票代码'].unique())
        stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
        
        # 使用LGBM的预处理
        processed, features = preprocess_data_lgbm(
            df, stockid2idx, self.config, is_train=False
        )
        
        # 获取最后一天的特征
        latest_date = processed['日期'].max()
        latest_data = processed[processed['日期'] == latest_date]
        
        if len(latest_data) == 0:
            raise ValueError("没有最新日期的数据")
        
        # 提取特征矩阵
        X = latest_data[features].values.astype(np.float32)
        stock_ids = latest_data['股票代码'].tolist()
        
        return {
            'X': X,
            'stock_ids': stock_ids,
            'features': features,
            'latest_date': latest_date
        }
    
    def prepare_validation_data(self):
        """准备验证集数据（用于权重优化）"""
        train_df, _ = self.load_raw_data()
        
        if train_df is None:
            raise ValueError("训练数据为空")
        
        # 按最后一个月划分验证集
        train_df = train_df.sort_values('日期')
        val_start_date = train_df['日期'].max() - pd.Timedelta(days=30)
        val_df = train_df[train_df['日期'] >= val_start_date]
        train_df = train_df[train_df['日期'] < val_start_date]
        
        if len(val_df) == 0:
            raise ValueError("验证集数据为空，请检查数据日期范围")
        
        print(f"训练集日期范围: {train_df['日期'].min()} 到 {train_df['日期'].max()}")
        print(f"验证集日期范围: {val_df['日期'].min()} 到 {val_df['日期'].max()}")
        
        # 为两个模型准备数据
        print("为Transformer准备验证数据...")
        transformer_data = self.prepare_for_transformer(val_df)
        
        print("为LGBM准备验证数据...")
        lgbm_data = self.prepare_for_lgbm(val_df)
        
        # 获取真实标签（T+1到T+5收益率）
        # 这里需要从验证数据中计算真实收益
        true_returns = self._extract_true_returns(val_df)
        
        return {
            'transformer': transformer_data,
            'lgbm': lgbm_data,
            'true_returns': true_returns,
            'dates': val_df['日期'].unique()
        }
    
    def _extract_true_returns(self, df):
        """从数据中提取真实收益率（T+1到T+5）"""
        # 复制数据以避免修改原始数据
        df = df.copy()
        
        # 按股票分组
        returns = []
        stock_ids = []
        
        for stock_id, group in df.groupby('股票代码'):
            group = group.sort_values('日期')
            
            # 计算T+1到T+5收益率
            group['open_t1'] = group['开盘'].shift(-1)
            group['open_t5'] = group['开盘'].shift(-5)
            group['return'] = (group['open_t5'] - group['open_t1']) / (group['open_t1'] + 1e-12)
            
            # 只保留有收益率的行
            valid_returns = group['return'].dropna()
            if len(valid_returns) > 0:
                # 取最后一天（最新）的收益率作为验证标签
                last_return = valid_returns.iloc[-1]
                returns.append(last_return)
                stock_ids.append(stock_id)
        
        print(f"提取了 {len(returns)} 只股票的真实收益率")
        return np.array(returns)
    
    def load_scaler(self):
        """加载Transformer的标准化器"""
        scaler_path = self.config['model_paths']['transformer']['scaler_path']
        if not scaler_path.exists():
            raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
        
        return joblib.load(scaler_path)


def create_flat_dataset_for_ensemble(data, features, sequence_length):
    """
    为集成模型创建扁平化数据集（用于训练元模型）
    
    Args:
        data: 预处理后的DataFrame
        features: 特征列名列表
        sequence_length: 序列长度
        
    Returns:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签向量 (n_samples,)
        dates: 日期向量
        stock_ids: 股票ID向量
    """
    print("正在为集成模型创建扁平化数据集...")
    
    # 确保数据按股票和时间排序
    data = data.sort_values(['股票代码', '日期']).reset_index(drop=True)
    
    X_list, y_list, date_list, stock_list = [], [], [], []
    
    # 按股票分组处理
    for stock_id, group in data.groupby('股票代码'):
        group = group.sort_values('日期')
        
        # 滑动窗口创建样本
        for i in range(len(group) - sequence_length - 5):  # -5保证有T+5标签
            # 提取序列
            sequence = group.iloc[i:i+sequence_length]
            
            # 提取最后一天的特征
            last_day_features = sequence[features].iloc[-1].values.astype(np.float32)
            
            # 获取标签（T+1到T+5收益率）
            label = group.iloc[i+sequence_length-1]['label']  # 假设label列已存在
            
            X_list.append(last_day_features)
            y_list.append(label)
            date_list.append(group.iloc[i+sequence_length-1]['日期'])
            stock_list.append(stock_id)
    
    if len(X_list) == 0:
        raise ValueError("未创建任何样本，请检查数据长度和序列长度")
    
    X = np.array(X_list)
    y = np.array(y_list)
    dates = np.array(date_list)
    stock_ids = np.array(stock_list)
    
    print(f"创建数据集: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    return X, y, dates, stock_ids