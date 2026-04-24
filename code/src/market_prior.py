import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'


class MarketStateExtractor:
    def __init__(self, index_csv='index_data.csv'):
        index_path = Path(index_csv)
        self.index_path = index_path if index_path.is_absolute() else DATA_DIR / index_csv
        self._load_and_prepare()

    def _load_and_prepare(self):
        df = pd.read_csv(self.index_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)

        ret_20d = df['close'].pct_change(20)
        vol_20d = df['volatility_20d']

        med_vol = vol_20d.median()

        conditions = [
            (ret_20d > 0.03) & (vol_20d <= med_vol),
            (ret_20d < -0.03) & (vol_20d <= med_vol),
            (vol_20d > med_vol),
        ]
        choices = [0, 2, 3]
        states = np.select(conditions, choices, default=1)

        self._data = pd.DataFrame({
            'date': df['date'],
            'state': states,
            'ret_20d': ret_20d,
            'vol_20d': vol_20d,
        }).dropna(subset=['state']).reset_index(drop=True)
        self._state_map = {0: 'bull', 1: 'range', 2: 'bear', 3: 'rotate'}

    def get_state(self, date_str):
        mask = self._data['date'] <= pd.Timestamp(date_str)
        if not mask.any():
            return 1
        row = self._data.loc[mask].iloc[-1]
        return int(row['state'])

    def get_state_onehot(self, date_str):
        s = self.get_state(date_str)
        oh = np.zeros(4, dtype=np.float32)
        oh[s] = 1.0
        return oh

    def get_volatility(self, date_str):
        mask = self._data['date'] <= pd.Timestamp(date_str)
        if not mask.any():
            return 0.0
        row = self._data.loc[mask].iloc[-1]
        return float(row['vol_20d']) if pd.notna(row['vol_20d']) else 0.0


class IndustryPriorComputer:
    def __init__(self, industry_csv='stock_industry.csv',
                 stock_data_csv='stock_data.csv',
                 lookback=5):
        ind_path = Path(industry_csv)
        self.industry_path = ind_path if ind_path.is_absolute() else DATA_DIR / industry_csv
        stock_path = Path(stock_data_csv)
        self.stock_data_path = stock_path if stock_path.is_absolute() else DATA_DIR / stock_data_csv
        self.lookback = lookback
        self._load()

    @staticmethod
    def _normalize_code(code):
        code = str(code).strip()
        if '.' in code:
            code = code.split('.')[1]
        return code

    def _load(self):
        self.industry_df = pd.read_csv(self.industry_path)
        self.industry_df['code'] = self.industry_df['code'].astype(str).str.strip()

        self.industry_map = {}
        for _, row in self.industry_df.iterrows():
            code = self._normalize_code(row['code'])
            ind = str(row.get('industry', 'Unknown')).strip()
            if ind == 'nan' or ind == '':
                ind = 'Unknown'
            self.industry_map[code] = ind

        self.stock_df = pd.read_csv(self.stock_data_path, parse_dates=['日期'])
        self.stock_df = self.stock_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
        self.stock_df['股票代码'] = self.stock_df['股票代码'].astype(str).str.strip()

        self.stock_df['industry'] = self.stock_df['股票代码'].map(self.industry_map).fillna('Unknown')
        self.stock_df['return_1d'] = self.stock_df.groupby('股票代码')['收盘'].pct_change()

        pivot = self.stock_df.pivot_table(
            index='日期', columns='industry', values='return_1d', aggfunc='mean'
        )
        pivot = pivot.rolling(window=self.lookback, min_periods=1).mean()
        self.industry_return_pivot = pivot

        self.all_dates = sorted(self.stock_df['日期'].unique())

    def get_prior_returns(self, date_str, stock_codes):
        dates = pd.to_datetime(self.all_dates)
        target = pd.Timestamp(date_str)
        mask = dates <= target
        if not mask.any():
            return np.zeros(len(stock_codes), dtype=np.float32)

        latest_date = self.industry_return_pivot.index[
            self.industry_return_pivot.index <= pd.Timestamp(date_str)
        ]
        if len(latest_date) == 0:
            return np.zeros(len(stock_codes), dtype=np.float32)
        latest_date = latest_date[-1]
        row = self.industry_return_pivot.loc[latest_date]

        prior_list = []
        for code in stock_codes:
            code = self._normalize_code(code)
            ind = self.industry_map.get(code, 'Unknown')
            val = row.get(ind, np.nan)
            prior_list.append(0.0 if pd.isna(val) else val)

        return np.array(prior_list, dtype=np.float32)


class AdaptiveWeightNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, market_state_onehot, vol_scalar):
        x = torch.cat([market_state_onehot, vol_scalar.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)
