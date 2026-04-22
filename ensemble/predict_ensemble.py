"""
集成模型预测脚本
加载集成模型，预测top5股票
"""
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'LGBM'))

from ensemble.config import ENSEMBLE_CONFIG
from ensemble.data_loader import EnsembleDataLoader
from ensemble.ensemble_model import EnsembleModel


def main():
    """主函数：集成模型预测"""
    parser = argparse.ArgumentParser(description='集成模型预测')
    parser.add_argument('--config', type=str, default='config.py',
                       help='配置文件路径')
    parser.add_argument('--model-path', type=str, 
                       help='集成模型路径，如果未指定则使用最佳模型')
    parser.add_argument('--score-format', action='store_true',
                       help='输出评分兼容格式')
    parser.add_argument('--team-name', type=str, default='ensemble_model',
                       help='团队名称，用于评分')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='输出目录')
    parser.add_argument('--verbose', action='store_true',
                       help='输出详细预测信息')
    args = parser.parse_args()
    
    print("=" * 60)
    print("集成模型预测")
    print("=" * 60)
    
    # 加载配置
    config = ENSEMBLE_CONFIG
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("\n1. 加载集成模型...")
    if args.model_path:
        # 从指定路径加载
        try:
            ensemble_model = EnsembleModel.load(args.model_path, config)
            print(f"从指定路径加载模型: {args.model_path}")
        except Exception as e:
            print(f"从指定路径加载模型失败: {e}")
            print("尝试从配置加载模型...")
            ensemble_model = EnsembleModel.from_config(config)
    else:
        # 加载最佳配置并创建模型
        ensemble_model = EnsembleModel.from_config(config)
    
    # 显示模型信息
    model_info = ensemble_model.get_model_info()
    print(f"  集成权重: Transformer={model_info['transformer_weight']:.3f}, LGBM={model_info['lgbm_weight']:.3f}")
    print(f"  Transformer设备: {model_info['transformer_device']}")
    print(f"  LGBM类型: {model_info['lgbm_type']}")
    
    # 准备数据
    print("\n2. 准备预测数据...")
    data_loader = EnsembleDataLoader(config)
    
    try:
        train_df, test_df = data_loader.load_raw_data()
    except Exception as e:
        print(f"加载数据失败: {e}")
        raise
    
    # 使用最新数据（训练集最后一天）
    latest_date = train_df['日期'].max()
    latest_data = train_df[train_df['日期'] == latest_date]
    
    if len(latest_data) == 0:
        raise ValueError("没有最新日期的数据")
    
    print(f"预测基准日: {latest_date}")
    print(f"可用股票数量: {len(latest_data)}")
    
    # 为两个模型准备数据
    print("  为Transformer准备数据...")
    transformer_data = data_loader.prepare_for_transformer(latest_data)
    
    print("  为LGBM准备数据...")
    lgbm_data = data_loader.prepare_for_lgbm(latest_data)
    
    print(f"  Transformer数据: {len(transformer_data['stock_ids'])} 只股票")
    print(f"  LGBM数据: {len(lgbm_data['stock_ids'])} 只股票")
    
    # 预测
    print("\n3. 进行集成预测...")
    try:
        predictions = ensemble_model.predict(transformer_data, lgbm_data)
    except Exception as e:
        print(f"集成预测失败: {e}")
        raise
    
    # 获取top5股票
    ensemble_scores = predictions['ensemble']
    stock_ids = predictions['stock_ids']
    
    sorted_indices = np.argsort(ensemble_scores)[::-1]  # 降序
    top_k = 5
    top_k = min(top_k, len(sorted_indices))
    top_indices = sorted_indices[:top_k]
    
    top_stocks = [stock_ids[i] for i in top_indices]
    top_scores = [ensemble_scores[i] for i in top_indices]
    
    # 打印结果
    print("\n预测结果:")
    print("-" * 60)
    print(f"{'排名':<6} {'股票代码':<10} {'集成分数':<12} {'Transformer分数':<15} {'LGBM分数':<12}")
    print("-" * 60)
    for i, idx in enumerate(top_indices, 1):
        stock = stock_ids[idx]
        ensemble_score = ensemble_scores[idx]
        transformer_score = predictions['transformer'][idx]
        lgbm_score = predictions['lgbm'][idx]
        print(f"{i:<6} {stock:<10} {ensemble_score:<12.6f} {transformer_score:<15.6f} {lgbm_score:<12.6f}")
    print("-" * 60)
    
    # 保存结果
    print("\n4. 保存结果...")
    
    if args.score_format:
        # 评分兼容格式
        output_path = output_dir / 'result.csv'
        results = []
        for stock in top_stocks:
            results.append({
                'stock_id': stock,
                'weight': 0.2
            })
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"评分兼容格式结果已保存: {output_path}")
        
        # 复制到评分目录
        score_dir = PROJECT_ROOT / 'test' / 'results_output'
        score_dir.mkdir(parents=True, exist_ok=True)
        score_path = score_dir / f'{args.team_name}.csv'
        
        shutil.copy2(output_path, score_path)
        print(f"结果已复制到: {score_path}")
        
        print(f"\n团队名称: {args.team_name}")
        print(f"评分文件: {score_path}")
        print(f"运行评分命令: python test/score_docker.py {args.team_name}")
    
    else:
        # 详细格式
        output_path = output_dir / 'ensemble_predictions.csv'
        results = []
        for i, idx in enumerate(top_indices, 1):
            stock = stock_ids[idx]
            ensemble_score = ensemble_scores[idx]
            transformer_score = predictions['transformer'][idx]
            lgbm_score = predictions['lgbm'][idx]
            
            results.append({
                '排名': i,
                '股票代码': stock,
                '集成分数': ensemble_score,
                'Transformer分数': transformer_score,
                'LGBM分数': lgbm_score,
                'Transformer权重': ensemble_model.weight,
                'LGBM权重': 1 - ensemble_model.weight
            })
        
        df = pd.DataFrame(results)
        df['预测日期'] = latest_date
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"详细预测结果已保存: {output_path}")
        
        # 保存完整预测结果（所有股票）
        full_output_path = output_dir / 'all_predictions.csv'
        full_results = []
        for i, idx in enumerate(sorted_indices):
            stock = stock_ids[idx]
            ensemble_score = ensemble_scores[idx]
            transformer_score = predictions['transformer'][idx]
            lgbm_score = predictions['lgbm'][idx]
            
            full_results.append({
                '排名': i + 1,
                '股票代码': stock,
                '集成分数': ensemble_score,
                'Transformer分数': transformer_score,
                'LGBM分数': lgbm_score
            })
        
        full_df = pd.DataFrame(full_results)
        full_df.to_csv(full_output_path, index=False, encoding='utf-8-sig')
        print(f"完整预测结果已保存: {full_output_path}")
    
    # 如果启用详细模式，输出更多信息
    if args.verbose:
        print("\n详细预测信息:")
        print(f"股票总数: {len(stock_ids)}")
        print(f"集成分数范围: [{ensemble_scores.min():.6f}, {ensemble_scores.max():.6f}]")
        print(f"Transformer分数范围: [{predictions['transformer'].min():.6f}, {predictions['transformer'].max():.6f}]")
        print(f"LGBM分数范围: [{predictions['lgbm'].min():.6f}, {predictions['lgbm'].max():.6f}]")
        
        # 计算相关性
        correlation = np.corrcoef(predictions['transformer'], predictions['lgbm'])[0, 1]
        print(f"Transformer与LGBM分数相关性: {correlation:.4f}")
    
    print("\n" + "=" * 60)
    print("预测完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()