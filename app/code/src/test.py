import os
import multiprocessing as mp
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from featurework import engineer_features_158plus39

FEATURE_COLUMNS_158PLUS39 = [
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

def preprocess_predict_data(df, stockid2idx):
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError('输入数据为空，无法预测')

    num_processes = min(10, mp.cpu_count())
    print('cpus:', mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(engineer_features_158plus39, groups), total=len(groups), desc='预测集特征工程'))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])

    return processed

def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
    sequences, sequence_stock_codes = [], []
    for stock_id in stock_ids:
        stock_history = data[
            (data['股票代码'] == stock_id) &
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            sequences.append(stock_history[features].values.astype(np.float32))
            sequence_stock_codes.append(stock_id)

    if len(sequences) == 0:
        raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

    return np.asarray(sequences, dtype=np.float32), sequence_stock_codes

def main():
    data_file = os.path.join(config['data_path'], 'train.csv')
    model_path = os.path.join(config['output_dir'], 'best_model.pth')
    scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
    output_path = os.path.join('/app/output', 'result.csv')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'未找到模型文件: {model_path}')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    latest_date = raw_df['日期'].max()

    stock_codes = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_codes)}
    idx2stockid = {v: k for k, v in stockid2idx.items()}

    processed = preprocess_predict_data(raw_df, stockid2idx)
    processed[FEATURE_COLUMNS_158PLUS39] = processed[FEATURE_COLUMNS_158PLUS39].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = joblib.load(scaler_path)
    processed[FEATURE_COLUMNS_158PLUS39] = scaler.transform(processed[FEATURE_COLUMNS_158PLUS39])

    sequence_length = config['sequence_length']
    sequences_np, sequence_stock_codes = build_inference_sequences(
        processed,
        FEATURE_COLUMNS_158PLUS39,
        sequence_length,
        stock_codes,
        latest_date,
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = StockTransformer(input_dim=len(FEATURE_COLUMNS_158PLUS39), config=config, num_stocks=len(stock_codes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    prior_bias = None
    market_state = None
    if config.get('use_mdrp', False):
        from market_prior import IndustryPriorComputer, MarketStateExtractor
        prior_computer = IndustryPriorComputer(
            industry_csv=os.path.join(config['data_path'], 'stock_industry.csv'),
            stock_data_csv=os.path.join(config['data_path'], 'stock_data.csv'),
            lookback=config.get('mdrp_lookback', 5),
        )
        date_str = latest_date.strftime('%Y-%m-%d')
        prior_returns = prior_computer.get_prior_returns(date_str, sequence_stock_codes)
        prior_bias = torch.from_numpy(prior_returns).float().to(device)

        state_extractor = MarketStateExtractor(
            index_csv=os.path.join(config['data_path'], 'index_data.csv')
        )
        onehot = state_extractor.get_state_onehot(date_str)
        vol = state_extractor.get_volatility(date_str)
        market_state = torch.from_numpy(
            np.concatenate([onehot, [vol]]).astype(np.float32)
        ).float().to(device).unsqueeze(0)

    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
        if prior_bias is not None:
            prior_bias = prior_bias.unsqueeze(0)
        scores = model(x, prior_bias=prior_bias, market_state=market_state).squeeze(0).detach().cpu().numpy()

        order = np.argsort(scores)[::-1]
        ranked_stock_codes = [sequence_stock_codes[i] for i in order]
        ranked_scores = scores[order]

    top_k = 5
    if len(ranked_stock_codes) < top_k:
        raise ValueError(f'可预测股票不足{top_k}只，当前仅有 {len(ranked_stock_codes)} 只')
    top5_ids = ranked_stock_codes[:top_k]
    top5_weights = np.array([0.2] * top_k)

    output_df = pd.DataFrame({
        'stock_id': top5_ids,
        'weight': top5_weights,
    })
    output_df.to_csv(output_path, index=False)

    print(f'预测日期: {latest_date.date()}')
    print(f'参与排序股票数: {len(ranked_stock_codes)}')
    print(f'等权权重: {top5_weights}')
    print(f'权重和: {sum(top5_weights):.4f}')
    print(f'结果已写入: {output_path}')

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()