"""
训练/优化集成权重
在验证集上寻找最优的集成权重
"""
import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

# 添加必要的路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ensemble.config import ENSEMBLE_CONFIG
from ensemble.data_loader import EnsembleDataLoader
from ensemble.ensemble_model import EnsembleModel


def calculate_final_score(predicted_scores, true_returns, top_k=5):
    """
    计算final_score
    
    Args:
        predicted_scores: 预测分数
        true_returns: 真实收益率
        top_k: 取前k个
        
    Returns:
        final_score
    """
    if len(predicted_scores) != len(true_returns):
        raise ValueError(f"预测分数长度({len(predicted_scores)})与真实收益长度({len(true_returns)})不一致")
    
    if len(predicted_scores) < top_k:
        raise ValueError(f"股票数量({len(predicted_scores)})少于top_k({top_k})")
    
    # 按预测分数排序
    sorted_indices = np.argsort(predicted_scores)[::-1]  # 降序
    top_indices = sorted_indices[:top_k]
    
    # 计算预测收益和
    pred_return_sum = np.sum(true_returns[top_indices])
    
    # 计算理论最大收益和
    sorted_true = np.sort(true_returns)[::-1]
    max_return_sum = np.sum(sorted_true[:top_k])
    
    # 计算随机收益和（平均）
    random_return_sum = np.mean(true_returns) * top_k
    
    # 计算final_score
    if max_return_sum - random_return_sum == 0:
        return 0.0
    
    final_score = (pred_return_sum - random_return_sum) / (max_return_sum - random_return_sum)
    return final_score


def optimize_weights(transformer_scores, lgbm_scores, true_returns, 
                    weight_range=(0.0, 1.0, 0.1)):
    """
    网格搜索最优集成权重
    
    Args:
        transformer_scores: Transformer预测分数
        lgbm_scores: LGBM预测分数
        true_returns: 真实收益率
        weight_range: 权重搜索范围 (start, stop, step)
        
    Returns:
        best_weight: 最优权重
        best_score: 最优final_score
        results: 所有权重的结果
    """
    start, stop, step = weight_range
    weights = np.arange(start, stop + step/2, step)  # 加step/2处理浮点误差
    
    best_score = -float('inf')
    best_weight = 0.5
    results = []
    
    print(f"搜索权重范围: {start} 到 {stop}, 步长: {step}")
    print(f"测试权重: {list(np.round(weights, 3))}")
    
    for weight in weights:
        # 标准化分数
        norm_transformer = (transformer_scores - np.mean(transformer_scores)) / (np.std(transformer_scores) + 1e-8)
        norm_lgbm = (lgbm_scores - np.mean(lgbm_scores)) / (np.std(lgbm_scores) + 1e-8)
        
        # 计算集成分数
        ensemble_scores = weight * norm_transformer + (1 - weight) * norm_lgbm
        
        # 计算final_score
        final_score = calculate_final_score(ensemble_scores, true_returns)
        
        results.append({
            'weight': float(weight),
            'final_score': float(final_score),
            'ensemble_scores_mean': float(np.mean(ensemble_scores)),
            'ensemble_scores_std': float(np.std(ensemble_scores))
        })
        
        print(f"  权重 {weight:.3f}: final_score = {final_score:.6f}")
        
        if final_score > best_score:
            best_score = final_score
            best_weight = weight
    
    return best_weight, best_score, results


def main():
    """主函数：优化集成权重"""
    parser = argparse.ArgumentParser(description='优化集成模型权重')
    parser.add_argument('--config', type=str, default='config.py',
                       help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default='./model',
                       help='输出目录')
    parser.add_argument('--weight-range', type=str, default='0.0,1.0,0.1',
                       help='权重搜索范围，格式: start,stop,step')
    parser.add_argument('--top-k', type=int, default=5,
                       help='计算final_score时取前k个股票')
    parser.add_argument('--force', action='store_true',
                       help='强制重新优化，覆盖现有结果')
    args = parser.parse_args()
    
    print("=" * 60)
    print("集成模型权重优化")
    print("=" * 60)
    
    # 加载配置
    config = ENSEMBLE_CONFIG
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已有优化结果
    best_config_path = output_dir / 'best_ensemble_config.json'
    if best_config_path.exists() and not args.force:
        print(f"发现已有优化结果: {best_config_path}")
        print("使用 --force 参数强制重新优化")
        
        with open(best_config_path, 'r') as f:
            best_config = json.load(f)
        
        print(f"最佳权重: {best_config['best_weight']:.3f}")
        print(f"最佳final_score: {best_config['best_final_score']:.6f}")
        return
    
    # 准备数据
    print("\n1. 准备验证集数据...")
    data_loader = EnsembleDataLoader(config)
    
    try:
        val_data = data_loader.prepare_validation_data()
    except Exception as e:
        print(f"准备验证数据失败: {e}")
        print("请确保Transformer和LGBM模型已训练，且数据文件存在")
        raise
    
    if val_data['true_returns'] is None or len(val_data['true_returns']) == 0:
        raise ValueError("未能提取真实收益率，请检查数据")
    
    print(f"验证集股票数量: {len(val_data['true_returns'])}")
    print(f"真实收益率范围: [{val_data['true_returns'].min():.4f}, {val_data['true_returns'].max():.4f}]")
    
    # 加载模型
    print("\n2. 加载模型...")
    try:
        ensemble_model = EnsembleModel.from_config(config)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保模型文件存在:")
        print(f"  Transformer: {config['model_paths']['transformer']['model_path']}")
        print(f"  LGBM: {config['model_paths']['lgbm']['model_path']}")
        raise
    
    # 获取模型预测
    print("\n3. 获取模型预测...")
    try:
        predictions = ensemble_model.predict(
            val_data['transformer'], 
            val_data['lgbm']
        )
    except Exception as e:
        print(f"模型预测失败: {e}")
        raise
    
    print(f"Transformer分数范围: [{predictions['transformer'].min():.4f}, {predictions['transformer'].max():.4f}]")
    print(f"LGBM分数范围: [{predictions['lgbm'].min():.4f}, {predictions['lgbm'].max():.4f}]")
    
    # 优化权重
    print("\n4. 优化集成权重...")
    weight_range = tuple(map(float, args.weight_range.split(',')))
    if len(weight_range) != 3:
        raise ValueError("weight-range参数格式错误，应为: start,stop,step")
    
    best_weight, best_score, results = optimize_weights(
        predictions['transformer'],
        predictions['lgbm'],
        val_data['true_returns'],
        weight_range
    )
    
    # 保存结果
    print("\n5. 保存结果...")
    results_df = pd.DataFrame(results)
    results_path = output_dir / 'weight_optimization.csv'
    results_df.to_csv(results_path, index=False)
    
    # 保存最佳配置
    best_config = {
        'best_weight': float(best_weight),
        'best_final_score': float(best_score),
        'optimization_date': pd.Timestamp.now().isoformat(),
        'weight_range': weight_range,
        'validation_size': len(val_data['true_returns']),
        'top_k': args.top_k,
        'transformer_score_mean': float(np.mean(predictions['transformer'])),
        'transformer_score_std': float(np.std(predictions['transformer'])),
        'lgbm_score_mean': float(np.mean(predictions['lgbm'])),
        'lgbm_score_std': float(np.std(predictions['lgbm']))
    }
    
    config_path = output_dir / 'best_ensemble_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    # 使用最佳权重创建最终模型
    ensemble_model.set_weight(best_weight)
    model_path = output_dir / 'ensemble_model.pkl'
    ensemble_model.save(model_path)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)
    print(f"最佳权重: {best_weight:.3f} (Transformer权重)")
    print(f"最佳final_score: {best_score:.6f}")
    print(f"权重搜索结果: {results_path}")
    print(f"最佳配置: {config_path}")
    print(f"集成模型配置: {model_path}")
    
    # 显示权重搜索结果
    print("\n权重搜索结果:")
    print("-" * 40)
    print(f"{'权重':<8} {'final_score':<12}")
    print("-" * 40)
    for result in results:
        weight = result['weight']
        score = result['final_score']
        marker = " ← 最佳" if abs(weight - best_weight) < 1e-6 else ""
        print(f"{weight:<8.3f} {score:<12.6f}{marker}")
    print("-" * 40)
    
    # 计算单一模型性能对比
    transformer_score = calculate_final_score(predictions['transformer'], val_data['true_returns'])
    lgbm_score = calculate_final_score(predictions['lgbm'], val_data['true_returns'])
    
    print(f"\n模型性能对比:")
    print(f"  Transformer单独: {transformer_score:.6f}")
    print(f"  LGBM单独:       {lgbm_score:.6f}")
    print(f"  集成模型:       {best_score:.6f}")
    print(f"  相比Transformer提升: {best_score - transformer_score:.6f}")
    print(f"  相比LGBM提升:       {best_score - lgbm_score:.6f}")


if __name__ == '__main__':
    main()