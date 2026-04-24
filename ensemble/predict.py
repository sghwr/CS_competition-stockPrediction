import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
_SRC = str(_PROJECT_ROOT / 'code' / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from .config import ENSEMBLE_CONFIG, FEATURE_COLUMNS
from .core import Ensemble


def main():
    cfg = ENSEMBLE_CONFIG
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # 1. 加载数据
    raw = pd.read_csv(cfg['stock_data'], dtype={'股票代码': str})
    raw['股票代码'] = raw['股票代码'].astype(str).str.strip()
    raw['日期'] = pd.to_datetime(raw['日期'])

    # 2. 特征工程
    ens = Ensemble(cfg)
    processed = ens.feature_engineer(raw)
    processed[FEATURE_COLUMNS] = processed[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 3. 加载 scaler（用于 transformer）
    scaler = ens.load_scaler()

    # 4. 取 test.csv 最新日期
    test = pd.read_csv(cfg['test_csv'])
    pred_date = pd.to_datetime(test['日期'].max())
    print(f'预测日期: {pred_date.date()}')

    # 5. 集成预测
    result = ens.predict_date(processed, FEATURE_COLUMNS, pred_date, scaler)
    if result is None:
        print('无有效预测')
        return

    scores, stock_ids = result['scores'], result['stock_ids']
    order = np.argsort(scores)[::-1]
    top_k = cfg['top_k']
    top_n = min(top_k, len(order))
    top_idx = order[:top_n]
    top_stocks = [stock_ids[i] for i in top_idx]

    weight = 1.0 / top_n
    weights = [weight] * top_n
    weights[-1] = 1.0 - weight * (top_n - 1)

    out_df = pd.DataFrame({'stock_id': top_stocks, 'weight': weights})
    out_path = os.path.join(cfg['output_dir'], 'result.csv')
    out_df.to_csv(out_path, index=False)

    print(f'选股: {top_stocks}')
    print(f'权重: {[round(w, 4) for w in weights]}')
    print(f'结果保存: {out_path}')


if __name__ == '__main__':
    mp.freeze_support()
    import multiprocessing as mp
    main()
