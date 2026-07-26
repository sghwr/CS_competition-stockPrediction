# 配置参数
sequence_length = 60
feature_num = '158+39'

# ============================================================================
# 数据划分 (Train/Val/Test) — 严格无重叠, 无空窗
#   - Train: 训练 base models (Tree / Linear / Transformer), 10 年
#   - Val:   验证, 12 月, 用于早停
#           (Stack 训在 TRAIN, VAL 仅用于 OOS 评估)
#   - Test:  终极 OOS, 6 月, 严格评估
# 数据源: data/stock_data.csv (2015-01-05 ~ 2026-06-26, 300 股)
# ============================================================================
SPLITS = {
    'train_start':  '2015-01-05',
    'train_end':    '2024-12-31',  # Train 10 年
    'val_start':    '2025-01-01',  # Val: 12 月 (1 年)
    'val_end':      '2025-12-31',
    'test_start':   '2026-01-01',  # Test: 终极 OOS (6 月)
    'test_end':     '2026-06-26',
}

config = {
    'sequence_length': sequence_length,
    'd_model': 256,
    'nhead': 4,
    'num_layers': 3,
    'dim_feedforward': 512,
    'batch_size': 4,
    'max_epochs': 50,             # Transformer 最大 epoch (实际由早停决定)
    'early_stop_patience': 10,    # val_score 连续 N epoch 不提升则停止
    'learning_rate': 1e-5,
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 2.0,

    'output_dir': './model',
    'data_path': './',
    'temperature': 0.5,

    # --- 数据划分 (Train/Val/Test, 严格无重叠) ---
    'train_start':  SPLITS['train_start'],
    'train_end':    SPLITS['train_end'],
    'val_start':    SPLITS['val_start'],
    'val_end':      SPLITS['val_end'],
    'test_start':   SPLITS['test_start'],
    'test_end':     SPLITS['test_end'],

    # --- 集成学习配置 ---
    'use_integrated': False,
    'integrated_output_dir': './output/integrated_v1',
    # Branch 1: RawSequenceTransformer
    'raw_features': ['开盘', '最高', '最低', '收盘', '成交量', '成交额', '换手率'],
    'nhead_raw': 8,
    'num_layers_raw': 3,
    'dim_feedforward_raw': 512,
    # Branch 2: LightGBM
    'lgb_n_estimators': 500,
    'lgb_learning_rate': 0.05,
    'lgb_num_leaves': 63,
    'lgb_max_depth': 8,
    'lgb_min_data_in_leaf': 50,
    'lgb_feature_fraction': 0.8,
    'lgb_bagging_fraction': 0.8,
    'lgb_bagging_freq': 5,
    'lgb_lambdarank_truncation_level': 10,
    # Ensemble
    'ensemble_search_step': 0.05,
}

