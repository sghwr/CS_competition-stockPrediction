sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,
    'd_model': 256,
    'nhead': 4,
    'num_layers': 3,
    'dim_feedforward': 512,
    'batch_size': 4,
    'num_epochs': 50,
    'learning_rate': 1e-5,
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 2.0,

    'output_dir': '/app/model',
    'data_path': '/app/data',
    'market_normalizer_thresholds': (-0.01, 0.01),
    'temperature': 0.5,

    'use_mdrp': True,
    'industry_data_path': '/app/data/stock_industry.csv',
    'index_data_path': "/app/data/index_data.csv",
    'num_industries': 28,
    'mdrp_lookback': 5,

    'winsorize_range': (0.01, 0.99),
    'label_smoothing': 0.05,
    'time_decay_half_life': 180,
    'vol_normalize_labels': True,

    'portfolio_temperature': 0.5,
    'max_weight_per_stock': 0.4,
}