import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import joblib

class MarketNormalizer:
    def __init__(self, thresholds: Tuple[float, float] = (-0.01, 0.01)):
        self.thresholds = thresholds
        self.state_stats: Dict[str, Dict[str, float]] = {}
        self.date_to_state: Dict[pd.Timestamp, str] = {}
        self.states = ['down', 'neutral', 'up']
        
    def fit(self, data: pd.DataFrame) -> 'MarketNormalizer':
        if 'datetime' not in data.columns or 'label' not in data.columns:
            raise ValueError("数据必须包含 'datetime' 和 'label' 列")
        
        data = data.copy()
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        daily_avg = data.groupby('datetime')['label'].mean()
        
        lower, upper = self.thresholds
        bins = [-np.inf, lower, upper, np.inf]
        state_labels = pd.cut(daily_avg, bins=bins, labels=self.states)
        
        self.date_to_state = dict(zip(daily_avg.index, state_labels))
        
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
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'MarketNormalizer':
        return joblib.load(path)