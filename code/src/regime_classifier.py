"""3-Class Regime Classifier (deterministic, no-leak).

基于 300 只股票 equal-weighted past 5d return 的简单规则分类.

Why 300-equal-weighted instead of HS300?
  - HS300 数据范围 2019-2026, stock 2015-2026
  - 2015-2018 (975 days) HS300 缺失, fallback sideways, regime 不准
  - 300-equal 跟 HS300 反转信号几乎一样 (差 0.01%), 相关 0.95
  - 300-equal 跟 stock 完全一致, 抽象层次匹配

3 classes:
  - bull (2):    past_5d > +1.5%  (5d ~0.3%/天, 强势上涨)
  - bear (0):    past_5d < -1.5% (5d ~-0.3%/天, 强势下跌)
  - sideways (1): |past_5d| <= 1.5% (震荡, 占多数)

Strategy mapping:
  - bull:    momentum continuation, 选 high beta stocks (信息/通信)
  - sideways: factor + macro, balanced
  - bear:    defensive / reversal, 选 low beta / 输家反弹 stocks

Usage:
  classifier = RegimeClassifier()
  regime_pred, regime_prob = classifier.predict(dates)
  # regime_pred: (T,) int [0=bear, 1=sideways, 2=bull]
  # regime_prob:  (T, 3) softmax-style prob
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import STOCK_DATA_CSV


REGIME_NAMES = ['bear', 'sideways', 'bull']
BULL_THRESHOLD = 0.015  # past_5d > +1.5%  -> bull
BEAR_THRESHOLD = -0.015  # past_5d < -1.5%  -> bear

STRATEGY_MAPPING = {
    0: 'bear: defensive / reversal, 选 low beta + 输家反弹',
    1: 'sideways: factor + macro, balanced, 行业轮动',
    2: 'bull: momentum continuation, 选 high beta 行业 (信息/通信)',
}


class RegimeClassifier:
    """3-class regime classifier based on 300-equal-weighted past_5d return."""

    def __init__(self, stock_csv=STOCK_DATA_CSV,
                 bull_threshold=BULL_THRESHOLD,
                 bear_threshold=BEAR_THRESHOLD):
        self.stock_csv = stock_csv
        self.bull_thr = bull_threshold
        self.bear_thr = bear_threshold
        self.daily_market = None
        self._load_market_returns()

    def _load_market_returns(self):
        """Precompute 300-equal-weighted daily returns and past_5d.

        等权 (equal-weighted) 而非 mcap-weighted:
          - 等权更稳定, 不受市值加权干扰
          - 跟 stock 完全一致, 抽象层次匹配
          - mcap proxy (close*volume) 跟真实 mcap 不同, 容易有 bias
        """
        df = pd.read_csv(self.stock_csv, parse_dates=['日期'])
        df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
        df['ret_1d'] = df.groupby('股票代码')['收盘'].pct_change()
        # 300-equal-weighted daily market return
        daily = df.groupby('日期')['ret_1d'].mean().reset_index()
        daily = daily.sort_values('日期').reset_index(drop=True)
        daily['_d_str'] = daily['日期'].dt.strftime('%Y-%m-%d')
        # past 5d = sum of 5 daily returns (equivalently product-1)
        daily['past_5d'] = daily['ret_1d'].rolling(5).sum()
        self.daily_market = daily

    def predict(self, dates):
        """Predict regime for given dates.

        Args:
            dates: list of date strings 'YYYY-MM-DD'

        Returns:
            regime_pred: (T,) int [0=bear, 1=sideways, 2=bull]
            regime_prob:  (T, 3) float (one-hot)
        """
        T = len(dates)
        regime_pred = np.ones(T, dtype=np.int32)  # default sideways
        for i, d in enumerate(dates):
            row = self.daily_market[self.daily_market['_d_str'] == d]
            if len(row) == 0:
                continue
            p5 = row['past_5d'].values[0]
            if pd.isna(p5):
                continue
            if p5 > self.bull_thr:
                regime_pred[i] = 2
            elif p5 < self.bear_thr:
                regime_pred[i] = 0
            else:
                regime_pred[i] = 1

        regime_prob = np.zeros((T, 3), dtype=np.float32)
        for i in range(T):
            regime_prob[i, regime_pred[i]] = 1.0
        return regime_pred, regime_prob

    def distribution(self, dates):
        """Show regime distribution for dates."""
        regime_pred, _ = self.predict(dates)
        from collections import Counter
        c = Counter(regime_pred.tolist())
        return {REGIME_NAMES[k]: v for k, v in sorted(c.items())}


if __name__ == '__main__':
    print('Testing RegimeClassifier (300-equal-weighted) ...', flush=True)
    rc = RegimeClassifier()
    print(f'\nFull range: {rc.daily_market["_d_str"].iloc[0]} ~ '
          f'{rc.daily_market["_d_str"].iloc[-1]}', flush=True)
    print(f'Total days: {len(rc.daily_market):,}', flush=True)

    dates_all = rc.daily_market.dropna(subset=['past_5d'])['_d_str'].tolist()
    print(f'\nRegime distribution (all data, {len(dates_all)} days):')
    for name, count in rc.distribution(dates_all).items():
        pct = count / len(dates_all) * 100
        print(f'  {name:10s}: {count:>4} ({pct:.1f}%) | {STRATEGY_MAPPING[REGIME_NAMES.index(name)]}')

    # 训练/验证/测试 段
    print('\nRegime by SPLITS:')
    from config import SPLITS
    split_keys = [
        ('train', SPLITS['train_start'], SPLITS['train_end']),
        ('val', SPLITS['val_start'], SPLITS['val_end']),
        ('test', SPLITS['test_start'], SPLITS['test_end']),
    ]
    for split_name, s, e in split_keys:
        d_list = [d for d in dates_all if s <= d <= e]
        if not d_list:
            continue
        print(f'\n  {split_name} ({s} ~ {e}, {len(d_list)} days):')
        for name, count in rc.distribution(d_list).items():
            pct = count / len(d_list) * 100
            print(f'    {name:10s}: {count:>3} ({pct:.1f}%)')
