import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).parent.parent
_SRC = str(_PROJECT_ROOT / 'code' / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from .config import ENSEMBLE_CONFIG, FEATURE_COLUMNS
from .core import Ensemble


def get_holding_return(stock_code, pred_date, raw_df, hold=5):
    future = raw_df[
        (raw_df['股票代码'] == stock_code) & (raw_df['日期'] > pred_date)
    ].sort_values('日期').head(hold)
    if len(future) < hold:
        return None
    first_open = future.iloc[0]['开盘']
    last_open = future.iloc[-1]['开盘']
    return (last_open - first_open) / first_open


def compute_statistics(returns):
    returns = np.array(returns, dtype=np.float64)
    n = len(returns)
    if n == 0:
        return {'总窗口数': 0}
    avg = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    cum = float(np.prod(1 + returns) - 1)
    sharpe = float(np.sqrt(252) * avg / std) if std > 0 else 0.0
    win = float(np.mean(returns > 0))
    pos = returns[returns > 0]
    neg = returns[returns < 0]
    pl = float(pos.mean() / abs(neg.mean())) if len(neg) and neg.mean() != 0 else float('inf')
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    mdd = float(np.min(drawdown))
    return {
        '总窗口数': n,
        '平均收益率': round(avg, 6),
        '收益率标准差': round(std, 6),
        '年化Sharpe': round(sharpe, 4),
        '胜率': round(win, 4),
        '盈亏比': round(pl, 4),
        '最大回撤': round(mdd, 6),
        '累计收益率': round(cum, 6),
        '最大单期收益': round(float(np.max(returns)), 6),
        '最小单期收益': round(float(np.min(returns)), 6),
        '中位数收益': round(float(np.median(returns)), 6),
    }


def main():
    cfg = ENSEMBLE_CONFIG
    output_dir = os.path.join(cfg['output_dir'], 'eval')
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    raw = pd.read_csv(cfg['stock_data'], dtype={'股票代码': str})
    raw['股票代码'] = raw['股票代码'].astype(str).str.strip()
    raw['日期'] = pd.to_datetime(raw['日期'])

    # 2. 特征工程
    ens = Ensemble(cfg)
    processed = ens.feature_engineer(raw)
    processed[FEATURE_COLUMNS] = processed[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 3. 加载 scaler
    scaler = ens.load_scaler()

    # 4. 确定评估日期
    test = pd.read_csv(cfg['test_csv'])
    test_dates = sorted(pd.to_datetime(test['日期'].unique()))
    all_dates = sorted(processed['日期'].unique())
    seq_len = cfg['sequence_length']
    min_feasible = all_dates[seq_len - 1]
    eval_dates = [d for d in test_dates if min_feasible <= d <= all_dates[-1]]
    print(f'评估日期: {eval_dates[0].date()} ~ {eval_dates[-1].date()}, 共 {len(eval_dates)} 天')

    # 5. 逐日滑动
    results = []
    pbar = tqdm(eval_dates, desc='Ensemble 滑动评估')
    for pred_date in pbar:
        date_str = pred_date.strftime('%Y-%m-%d')
        out = ens.predict_date(processed, FEATURE_COLUMNS, pred_date, scaler)
        if out is None:
            continue
        scores, stock_ids = out['scores'], out['stock_ids']
        order = np.argsort(scores)[::-1]
        top_k = min(cfg['top_k'], len(order))
        top_idx = order[:top_k]
        top_stocks = [stock_ids[i] for i in top_idx]
        weight = 1.0 / top_k

        rets, valid_sids, valid_ws = [], [], []
        for sid in top_stocks:
            r = get_holding_return(sid, pred_date, raw, cfg['holding_days'])
            if r is not None:
                rets.append(r)
                valid_sids.append(sid)
                valid_ws.append(weight)
        if not rets:
            continue
        rets, valid_ws = np.array(rets), np.array(valid_ws)
        valid_ws /= valid_ws.sum()
        window_ret = float(np.sum(rets * valid_ws))

        row = {'日期': date_str}
        for k in range(top_k):
            row[f'stock_{k+1}'] = valid_sids[k] if k < len(valid_sids) else ''
            row[f'weight_{k+1}'] = round(valid_ws[k], 4) if k < len(valid_ws) else 0.0
        row['加权收益率'] = round(window_ret, 6)
        results.append(row)
        pbar.set_postfix({'收益': f'{window_ret:.4%}'})

    if not results:
        print('未产生有效窗口')
        return

    # 6. 结果
    result_df = pd.DataFrame(results)
    stats = compute_statistics(result_df['加权收益率'].values)

    stat_items = list(stats.items())
    stat_pairs = []
    for i in range(0, len(stat_items), 2):
        k1, v1 = stat_items[i]
        k2, v2 = stat_items[i + 1] if i + 1 < len(stat_items) else ('', '')
        stat_pairs.append((f'{k1}: {v1}', f'{k2}: {v2}'))

    top_k = cfg['top_k']
    stats_row_data = {'日期': '===== 统计汇总 ====='}
    for k in range(top_k):
        if k < len(stat_pairs):
            stats_row_data[f'stock_{k+1}'] = stat_pairs[k][0]
            stats_row_data[f'weight_{k+1}'] = stat_pairs[k][1]
        else:
            stats_row_data[f'stock_{k+1}'] = ''
            stats_row_data[f'weight_{k+1}'] = ''
    stats_row_data['加权收益率'] = ''

    stats_row = pd.DataFrame([stats_row_data])
    final_df = pd.concat([result_df, stats_row], ignore_index=True)
    csv_path = os.path.join(output_dir, 'ensemble_results.csv')
    final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f'\n结果保存: {csv_path}')
    print('\n======== Ensemble 统计汇总 ========')
    for k, v in stats.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()
