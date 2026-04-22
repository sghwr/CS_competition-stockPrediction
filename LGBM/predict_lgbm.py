"""
LGBM模型预测
加载训练好的LGBM模型，预测top5股票
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 添加LGBM目录到sys.path，确保优先导入LGBM/config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "code", "src"))

# 导入LGBM配置和数据加载器
from lgbm_config import lgbm_config
from data_loader import load_and_prepare_data, feature_cloums_map, feature_engineer_func_map


def load_lgbm_model(model_dir=None):
    """
    加载LGBM模型和配置
    
    Args:
        model_dir: 模型目录，如果为None则使用config中的output_dir
    
    Returns:
        model: LGBM模型
        config: 配置字典
        feature_columns: 特征列名列表
    """
    if model_dir is None:
        model_dir = lgbm_config['output_dir']
    
    # 加载配置
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 加载模型
    model_path = os.path.join(model_dir, 'lgbm_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = joblib.load(model_path)
    print(f"模型已加载: {model_path}")
    
    # 获取特征列
    feature_num = config['feature_num']
    feature_columns = feature_cloums_map[feature_num]
    
    return model, config, feature_columns


def prepare_prediction_data(data_dict, config, prediction_date=None):
    """
    准备预测数据
    
    Args:
        data_dict: load_and_prepare_data返回的数据字典
        config: 配置字典
        prediction_date: 预测日期，如果为None则使用最新日期
    
    Returns:
        X_pred: 特征矩阵 (n_stocks, n_features)
        stock_ids: 股票ID列表
        stock_indices: 股票索引列表
    """
    # 获取训练数据的最新日期
    train_df = data_dict['train_df']
    if prediction_date is None:
        prediction_date = train_df['日期'].max()
    
    print(f"预测基准日: {prediction_date}")
    
    # 获取所有股票ID
    all_stock_ids = sorted(train_df['股票代码'].unique())
    stockid2idx = data_dict['stockid2idx']
    
    # 准备特征矩阵
    X_pred = []
    valid_stock_ids = []
    valid_stock_indices = []
    
    feature_columns = data_dict['feature_columns']
    sequence_length = config['sequence_length']
    
    for stock_id in all_stock_ids:
        # 获取该股票的历史数据
        stock_data = train_df[train_df['股票代码'] == stock_id].copy()
        stock_data = stock_data.sort_values('日期')
        
        # 检查数据是否足够
        if len(stock_data) < sequence_length:
            continue
        
        # 获取最后sequence_length天的数据
        stock_history = stock_data.tail(sequence_length)
        
        # 检查是否包含预测日（或之前）的数据
        last_date = stock_history['日期'].iloc[-1]
        if pd.to_datetime(last_date) < pd.to_datetime(prediction_date):
            # 如果最新数据早于预测日，可能数据有延迟，跳过
            continue
        
        # 特征工程
        feature_engineer = feature_engineer_func_map[config['feature_num']]
        processed = feature_engineer(stock_history)
        
        # 提取最后一天的特征
        last_day_features = processed[feature_columns].iloc[-1].values.astype(np.float32)
        
        X_pred.append(last_day_features)
        valid_stock_ids.append(stock_id)
        valid_stock_indices.append(stockid2idx[stock_id])
    
    if len(X_pred) == 0:
        raise ValueError("没有可用的股票数据用于预测")
    
    X_pred = np.array(X_pred)
    print(f"准备预测数据: {X_pred.shape[0]} 只股票, {X_pred.shape[1]} 个特征")
    
    return X_pred, valid_stock_ids, valid_stock_indices


def predict_top_stocks(model, X_pred, stock_ids, top_k=5):
    """
    预测top_k股票
    
    Args:
        model: LGBM模型
        X_pred: 特征矩阵
        stock_ids: 股票ID列表
        top_k: 返回的股票数量
    
    Returns:
        top_stocks: 排序后的股票列表，每个元素为字典
    """
    # 预测分数
    scores = model.predict(X_pred)
    
    # 按分数排序
    sorted_indices = np.argsort(scores)[::-1]  # 降序
    top_indices = sorted_indices[:top_k]
    
    # 构建结果
    top_stocks = []
    for i, idx in enumerate(top_indices):
        stock_info = {
            'rank': i + 1,
            'stock_code': stock_ids[idx],
            'score': float(scores[idx]),
            'score_rank': int(sorted_indices.tolist().index(idx) + 1)
        }
        top_stocks.append(stock_info)
    
    return top_stocks


def save_predictions(top_stocks, output_path, prediction_date=None, format_type='detailed'):
    """
    保存预测结果
    
    Args:
        top_stocks: 预测结果列表
        output_path: 输出文件路径
        prediction_date: 预测日期
        format_type: 输出格式，'detailed'（详细格式）或'score'（评分兼容格式）
    """
    if format_type == 'score':
        # 评分兼容格式：两列，stock_id 和 weight（固定0.2）
        results = []
        for stock in top_stocks:
            results.append({
                'stock_id': stock['stock_code'],
                'weight': 0.2
            })
        df = pd.DataFrame(results)
        # 评分脚本要求列名为'stock_id'和'weight'
        df.to_csv(output_path, index=False)
        print(f"评分兼容格式结果已保存: {output_path}")
        return df
    else:
        # 详细格式（默认）
        results = []
        for stock in top_stocks:
            results.append({
                '排名': stock['rank'],
                '股票代码': stock['stock_code'],
                '预测分数': stock['score'],
                '分数排名': stock['score_rank']
            })
        
        df = pd.DataFrame(results)
        
        # 添加预测日期信息
        if prediction_date:
            df['预测日期'] = prediction_date
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"详细预测结果已保存: {output_path}")
        
        # 同时保存JSON格式
        json_path = output_path.replace('.csv', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'prediction_date': prediction_date,
                'top_stocks': top_stocks
            }, f, indent=2, ensure_ascii=False)
        
        return df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LGBM模型预测')
    parser.add_argument('--score-format', action='store_true', 
                       help='输出评分兼容格式到./output/result.csv')
    parser.add_argument('--team-name', type=str, default='lgbm_model',
                       help='团队名称，用于评分（默认: lgbm_model）')
    args = parser.parse_args()
    
    print("=" * 60)
    print("LGBM模型预测")
    if args.score_format:
        print("模式: 评分兼容格式")
    else:
        print("模式: 详细格式")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    model, config, feature_columns = load_lgbm_model()
    
    # 2. 加载数据
    print("\n2. 加载数据...")
    data_dict = load_and_prepare_data(config)
    
    # 3. 准备预测数据
    print("\n3. 准备预测数据...")
    X_pred, stock_ids, stock_indices = prepare_prediction_data(data_dict, config)
    
    # 4. 预测top5股票
    print("\n4. 预测top5股票...")
    top_stocks = predict_top_stocks(model, X_pred, stock_ids, top_k=5)
    
    # 5. 打印结果
    print("\n预测结果:")
    print("-" * 60)
    print(f"{'排名':<6} {'股票代码':<10} {'预测分数':<12} {'分数排名':<10}")
    print("-" * 60)
    for stock in top_stocks:
        print(f"{stock['rank']:<6} {stock['stock_code']:<10} {stock['score']:<12.6f} {stock['score_rank']:<10}")
    print("-" * 60)
    
    # 6. 保存结果
    print("\n5. 保存结果...")
    
    if args.score_format:
        # 评分兼容格式：输出到./output/result.csv
        output_dir = os.path.join(project_root, 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'result.csv')
        prediction_date = datetime.now().strftime('%Y-%m-%d')
        
        # 保存评分兼容格式
        save_predictions(top_stocks, output_path, prediction_date, format_type='score')
        
        # 同时复制到./test/results_output/{team_name}.csv
        results_output_dir = os.path.join(project_root, 'test', 'results_output')
        os.makedirs(results_output_dir, exist_ok=True)
        team_output_path = os.path.join(results_output_dir, f'{args.team_name}.csv')
        import shutil
        shutil.copy2(output_path, team_output_path)
        print(f"结果已复制到: {team_output_path}")
        
        print(f"\n团队名称: {args.team_name}")
        print(f"评分文件: {team_output_path}")
        print("运行评分命令: python test/score_docker.py", args.team_name)
    else:
        # 详细格式：保持原有行为
        output_dir = config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用当前日期作为预测日期
        prediction_date = datetime.now().strftime('%Y-%m-%d')
        output_path = os.path.join(output_dir, f'predictions_{prediction_date}.csv')
        
        save_predictions(top_stocks, output_path, prediction_date, format_type='detailed')
    
    print("\n" + "=" * 60)
    print("预测完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()