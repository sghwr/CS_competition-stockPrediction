# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 50,       # 排序任务可能需要更多epochs
    'learning_rate': 1e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './',
    'market_normalizer_thresholds': (-0.01, 0.01),  # 市场状态划分阈值
    'temperature': 0.5,  # 损失函数温度参数

    # --- MD-SRP 配置 ---
    'use_mdrp': True,                       # 是否启用 MD-SRP 模块
    'industry_data_path': './data/stock_industry.csv',
    'index_data_path': "./data/index_data.csv",
    'num_industries': 28,                    # 申万一级行业数
    'mdrp_lookback': 5,                      # 行业动量回溯窗口（交易日）

    # --- 自适应标签工程 ---
    'winsorize_range': (0.01, 0.99),         # 标签截尾范围（分位数）
    'label_smoothing': 0.05,                 # 标签平滑系数
    'time_decay_half_life': 180,             # 样本时间衰减半衰期（天）
    'vol_normalize_labels': True,            # 是否用波动率标准化标签

    # --- 动态组合 ---
    'portfolio_temperature': 0.5,            # Softmax 温度参数
    'max_weight_per_stock': 0.4,             # 单只股票最大权重
}