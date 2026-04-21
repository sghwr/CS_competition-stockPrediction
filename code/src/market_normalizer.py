import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import joblib

class MarketNormalizer:
    """
    分层市场标准化器
    根据每日市场平均收益率划分市场状态，对每个状态独立标准化收益率
    """
    
    def __init__(self, thresholds: Tuple[float, float] = (-0.01, 0.01)):
        """
        Args:
            thresholds: (lower, upper) 阈值，用于划分市场状态
                lower < 0, upper > 0
                收益率 < lower: 下跌市
                lower <= 收益率 <= upper: 震荡市  
                收益率 > upper: 上涨市
        """
        self.thresholds = thresholds
        self.state_stats: Dict[str, Dict[str, float]] = {}
        self.date_to_state: Dict[pd.Timestamp, str] = {}
        self.states = ['down', 'neutral', 'up']
        
    def fit(self, data: pd.DataFrame) -> 'MarketNormalizer':
        """
        基于历史数据拟合标准化器
        
        Args:
            data: DataFrame，必须包含列：'datetime', 'label'
        """
        if 'datetime' not in data.columns or 'label' not in data.columns:
            raise ValueError("数据必须包含 'datetime' 和 'label' 列")
        
        data = data.copy()
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # 计算每日市场平均收益率
        daily_avg = data.groupby('datetime')['label'].mean()
        
        # 根据阈值划分市场状态
        lower, upper = self.thresholds
        bins = [-np.inf, lower, upper, np.inf]
        state_labels = pd.cut(daily_avg, bins=bins, labels=self.states)
        
        self.date_to_state = dict(zip(daily_avg.index, state_labels))
        
        # 计算每个状态的统计量
        for state in self.states:
            state_dates = [date for date, s in self.date_to_state.items() if s == state]
            
            if not state_dates:
                self.state_stats[state] = {'mean': 0.0, 'std': 1.0}
                continue
                
            state_mask = data['datetime'].isin(state_dates)
            state_returns = data.loc[state_mask, 'label']
            
            mean = state_returns.mean()
            std = state_returns.std()
            
            if std < 1e-8:
                std = 1.0
                
            self.state_stats[state] = {'mean': mean, 'std': std}
        
        return self
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        标准化收益率
        
        Args:
            data: DataFrame，必须包含列：'datetime', 'label'
        
        Returns:
            normalized: 标准化后的收益率数组
        """
        if not self.state_stats:
            raise ValueError("必须先调用 fit 方法")
        
        data = data.copy()
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        normalized = np.zeros(len(data), dtype=np.float32)
        
        for i, row in data.iterrows():
            date = row['datetime']
            value = row['label']
            
            state = self.date_to_state.get(date, 'neutral')
            stats = self.state_stats[state]
            
            normalized[i] = (value - stats['mean']) / stats['std']
        
        return normalized
    
    def save(self, path: str) -> None:
        """保存标准化器到文件"""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'MarketNormalizer':
        """从文件加载标准化器"""
        return joblib.load(path)