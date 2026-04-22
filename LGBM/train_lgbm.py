"""
LGBM排序模型训练
使用LambdaRank进行股票排序，复用现有评估指标
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 添加LGBM目录到sys.path，确保优先导入LGBM/config
# 先添加code/src，然后添加LGBM目录，这样LGBM目录会在前面（后插入的在前）
sys.path.insert(0, os.path.join(project_root, "code", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入现有模块
from train import calculate_ranking_metrics, split_train_val_by_last_month

# 导入LGBM模块
try:
    import lightgbm as lgb
except ImportError:
    print("请安装lightgbm: pip install lightgbm")
    sys.exit(1)

# 导入LGBM配置和数据加载器
from lgbm_config import lgbm_config
from data_loader import load_and_prepare_data, create_flat_dataset_for_lgbm


def split_data_by_date(X, y, groups, dates, val_start_date):
    """
    按日期划分训练集和验证集
    
    Args:
        X: 特征矩阵
        y: 标签向量
        groups: 分组信息（每个日期的样本数）
        dates: 日期向量
        val_start_date: 验证集开始日期（datetime）
    
    Returns:
        (X_train, y_train, train_groups, train_dates),
        (X_val, y_val, val_groups, val_dates)
    """
    # 将日期转换为datetime
    if isinstance(dates[0], np.datetime64):
        dates_dt = pd.to_datetime(dates)
    else:
        dates_dt = pd.to_datetime(dates)
    
    # 创建mask
    train_mask = dates_dt < val_start_date
    val_mask = dates_dt >= val_start_date
    
    # 划分数据
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    train_dates, val_dates = dates[train_mask], dates[val_mask]
    
    # 重新计算分组信息
    train_date_counts = pd.Series(train_dates).value_counts().sort_index()
    val_date_counts = pd.Series(val_dates).value_counts().sort_index()
    
    train_groups = train_date_counts.values.tolist()
    val_groups = val_date_counts.values.tolist()
    
    print(f"训练集样本数: {len(X_train)}, 日期数: {len(train_groups)}")
    print(f"验证集样本数: {len(X_val)}, 日期数: {len(val_groups)}")
    
    return (X_train, y_train, train_groups, train_dates), (X_val, y_val, val_groups, val_dates)


def train_lgbm_ranking_model(config):
    """
    训练LGBM排序模型
    """
    print("=" * 60)
    print("LGBM排序模型训练")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    print(f"配置已保存: {config_path}")
    
    # 1. 加载和准备数据
    print("\n1. 加载和准备数据...")
    data_dict = load_and_prepare_data(config)
    
    # 2. 创建扁平化数据集
    print("\n2. 创建扁平化数据集...")
    X, y, groups, dates, stock_ids = create_flat_dataset_for_lgbm(
        data_dict['train_processed'],
        data_dict['feature_columns'],
        config['sequence_length'],
        market_normalizer=None  # 不使用市场标准化器
    )
    
    # 3. 按日期划分训练集和验证集
    print("\n3. 划分训练集和验证集...")
    
    # 获取验证集开始日期（最后一个月）
    all_dates = pd.to_datetime(dates)
    last_date = all_dates.max()
    val_start = last_date - pd.DateOffset(months=1)
    
    print(f"全量数据日期范围: {all_dates.min().date()} 到 {last_date.date()}")
    print(f"验证集开始日期: {val_start.date()}")
    
    # 划分数据
    (X_train, y_train, train_groups, train_dates), \
    (X_val, y_val, val_groups, val_dates) = split_data_by_date(
        X, y, groups, dates, val_start
    )
    
    # 4. 训练LGBM排序模型
    print("\n4. 训练LGBM排序模型...")
    
    # 准备LGBM数据集
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        group=train_groups,
        free_raw_data=False
    )
    
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        group=val_groups,
        reference=train_data,
        free_raw_data=False
    )
    
    # LGBM参数
    lgbm_params = {
        'objective': 'lambdarank',
        'metric': config.get('eval_metric', 'ndcg'),
        'ndcg_eval_at': [5],  # 评估Top5 NDCG
        'boosting_type': 'gbdt',
        'num_leaves': config['num_leaves'],
        'max_depth': config['max_depth'],
        'learning_rate': config['learning_rate'],
        'n_estimators': config['n_estimators'],
        'subsample': config['subsample'],
        'colsample_bytree': config['colsample_bytree'],
        'reg_alpha': config['reg_alpha'],
        'reg_lambda': config['reg_lambda'],
        'min_child_samples': config['min_child_samples'],
        'min_child_weight': config['min_child_weight'],
        'random_state': config['random_state'],
        'n_jobs': config['n_jobs'],
        'verbosity': -1,
        # label_gain: 映射标签值到增益值，长度必须至少为最大标签值+1
        # 我们的标签是1到298的排名，所以需要299个映射
        'label_gain': list(range(300)),  # 0-299，索引0未使用
    }
    
    print("LGBM参数:")
    for key, value in lgbm_params.items():
        if key not in ['verbosity']:
            print(f"  {key}: {value}")
    
    # 训练模型
    print("\n开始训练...")
    model = lgb.train(
        lgbm_params,
        train_data,
        valid_sets=[val_data],
        valid_names=['valid'],
        num_boost_round=config['n_estimators'],
        callbacks=[
            lgb.early_stopping(config['early_stopping_rounds']),
            lgb.log_evaluation(50)  # 每50轮打印一次
        ]
    )
    
    # 5. 保存模型
    print("\n5. 保存模型...")
    model_path = os.path.join(output_dir, 'lgbm_model.pkl')
    joblib.dump(model, model_path)
    print(f"模型已保存: {model_path}")
    
    # 保存特征重要性
    feature_importance = pd.DataFrame({
        'feature': data_dict['feature_columns'],
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')
    print(f"特征重要性已保存: {importance_path}")
    
    # 6. 评估模型
    print("\n6. 评估模型...")
    
    # 在验证集上预测
    y_pred_val = model.predict(X_val)
    
    # 按日期分组评估（复用现有评估逻辑）
    val_results = evaluate_lgbm_predictions(
        y_pred_val, y_val, val_groups, val_dates, stock_ids[dates >= val_start]
    )
    
    # 保存评估结果
    eval_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(val_results, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存: {eval_path}")
    
    # 7. 保存final_score
    final_score = val_results.get('final_score', 0.0)
    final_score_path = os.path.join(output_dir, 'final_score.txt')
    with open(final_score_path, 'w', encoding='utf-8') as f:
        f.write(f"Best final_score: {final_score:.6f}\n")
    print(f"Final score: {final_score:.6f}")
    print(f"Final score已保存: {final_score_path}")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"所有文件已保存到: {output_dir}")
    print("=" * 60)
    
    return model, val_results


def evaluate_lgbm_predictions(y_pred, y_true, groups, dates, stock_ids, k=5):
    """
    评估LGBM预测结果，复用现有的calculate_ranking_metrics逻辑
    
    Args:
        y_pred: 预测分数
        y_true: 真实标签
        groups: 分组信息（每个日期的样本数）
        dates: 日期向量
        stock_ids: 股票ID向量
        k: Top-k数量
    
    Returns:
        评估指标字典
    """
    # 将数据转换为与Transformer相同的格式
    # 按日期分组
    unique_dates = np.unique(dates)
    
    # 初始化累计指标
    all_pred_return_sum = []
    all_max_return_sum = []
    all_random_return_sum = []
    all_final_score = []
    
    # 对每个日期单独评估
    start_idx = 0
    for date_idx, date in enumerate(unique_dates):
        group_size = groups[date_idx]
        end_idx = start_idx + group_size
        
        # 提取当前日期的数据
        date_pred = y_pred[start_idx:end_idx]
        date_true = y_true[start_idx:end_idx]
        
        if len(date_pred) < k:
            start_idx = end_idx
            continue
        
        # 转换为torch tensor格式（模拟Transformer评估）
        pred_tensor = np.array(date_pred).reshape(1, -1)  # [1, num_stocks]
        true_tensor = np.array(date_true).reshape(1, -1)  # [1, num_stocks]
        
        # 创建mask（所有股票都有效）
        mask = np.ones_like(pred_tensor, dtype=bool)
        
        # 计算指标（这里简化，直接计算）
        # 按预测分数排序
        pred_indices = np.argsort(date_pred)[::-1][:k]
        true_indices = np.argsort(date_true)[::-1][:k]
        
        pred_return_sum = date_true[pred_indices].sum()
        max_return_sum = date_true[true_indices].sum()
        random_return_sum = k * date_true.mean()
        
        # 计算final_score
        denominator = max_return_sum - random_return_sum
        if abs(denominator) > 1e-6:
            final_score = (pred_return_sum - random_return_sum) / denominator
        else:
            final_score = 0.0
        
        all_pred_return_sum.append(pred_return_sum)
        all_max_return_sum.append(max_return_sum)
        all_random_return_sum.append(random_return_sum)
        all_final_score.append(final_score)
        
        start_idx = end_idx
    
    # 计算平均指标
    if len(all_final_score) > 0:
        results = {
            'pred_return_sum': np.mean(all_pred_return_sum),
            'max_return_sum': np.mean(all_max_return_sum),
            'random_return_sum': np.mean(all_random_return_sum),
            'final_score': np.mean(all_final_score),
            'num_dates_evaluated': len(all_final_score),
        }
    else:
        results = {
            'pred_return_sum': 0.0,
            'max_return_sum': 0.0,
            'random_return_sum': 0.0,
            'final_score': 0.0,
            'num_dates_evaluated': 0,
        }
    
    print(f"评估了 {results['num_dates_evaluated']} 个交易日")
    print(f"Predicted Top5 Return Sum: {results['pred_return_sum']:.6f}")
    print(f"Theoretical Max Return Sum: {results['max_return_sum']:.6f}")
    print(f"Random Return Sum: {results['random_return_sum']:.6f}")
    print(f"Final Score: {results['final_score']:.6f}")
    
    return results


def main():
    """主函数"""
    try:
        model, results = train_lgbm_ranking_model(lgbm_config)
        
        # 打印最终结果
        print("\n最终结果:")
        print(f"Final Score: {results['final_score']:.6f}")
        print(f"评估交易日数: {results['num_dates_evaluated']}")
        
        # 判断模型效果
        if results['final_score'] > 0.1:
            print("✅ 模型表现良好")
        elif results['final_score'] > 0:
            print("⚠️  模型表现一般")
        else:
            print("❌ 模型表现不佳，需要进一步优化")
            
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()