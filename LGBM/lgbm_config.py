# LGBM模型配置
import os
import sys
import json

# 添加项目根目录到路径，以便导入现有模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "code", "src"))

# 导入现有配置
import config as base_config_module
base_config = base_config_module.config

# LGBM专用配置
lgbm_config = {
    # 复用现有配置
    'sequence_length': base_config['sequence_length'],
    'feature_num': base_config['feature_num'],
    'data_path': os.path.join(project_root, 'data'),
    
    # 模型类型：'regression' 或 'ranking' (LambdaRank)
    'model_type': 'ranking',
    
    # LGBM超参数
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
    
    # 早停设置
    'early_stopping_rounds': 50,
    
    # 评估指标
    'eval_metric': 'ndcg',  # 排序任务使用NDCG
    
    # 输出目录
    'output_dir': os.path.join(project_root, 'LGBM', 'model'),
    
    # 标签计算模式：'t_to_t5' 或 't1_to_t5'
    'label_mode': 't1_to_t5',  # 使用 T+1 到 T+5 的收益率
    
    # 数据保存路径
    'train_ranking_data_path': None,  # 可设置为路径以缓存数据集
    'val_ranking_data_path': None,
}

# 确保输出目录存在
os.makedirs(lgbm_config['output_dir'], exist_ok=True)

if __name__ == "__main__":
    print("LGBM配置:")
    for key, value in lgbm_config.items():
        print(f"  {key}: {value}")