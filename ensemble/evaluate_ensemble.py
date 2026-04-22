"""
集成模型评估脚本
在验证集上评估集成模型性能
"""
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

from ensemble.config import ENSEMBLE_CONFIG
from ensemble.data_loader import EnsembleDataLoader
from ensemble.ensemble_model import EnsembleModel


def calculate_final_score(predicted_scores, true_returns, top_k=5):
    """计算final_score"""
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


def evaluate_ensemble(config, ensemble_model, val_data, top_k=5):
    """评估集成模型"""
    # 获取预测
    predictions = ensemble_model.predict(
        val_data['transformer'],
        val_data['lgbm']
    )
    
    # 计算各个模型的final_score
    transformer_score = calculate_final_score(
        predictions['transformer'], val_data['true_returns'], top_k
    )
    lgbm_score = calculate_final_score(
        predictions['lgbm'], val_data['true_returns'], top_k
    )
    ensemble_score = calculate_final_score(
        predictions['ensemble'], val_data['true_returns'], top_k
    )
    
    # 计算其他指标
    metrics = {
        'transformer_final_score': float(transformer_score),
        'lgbm_final_score': float(lgbm_score),
        'ensemble_final_score': float(ensemble_score),
        'improvement_vs_transformer': float(ensemble_score - transformer_score),
        'improvement_vs_lgbm': float(ensemble_score - lgbm_score),
        'improvement_percent_vs_transformer': float((ensemble_score - transformer_score) / (abs(transformer_score) + 1e-8) * 100),
        'improvement_percent_vs_lgbm': float((ensemble_score - lgbm_score) / (abs(lgbm_score) + 1e-8) * 100),
        'weight': float(ensemble_model.weight),
        'num_stocks': len(predictions['stock_ids']),
        'validation_dates': len(set(val_data['dates'])) if 'dates' in val_data else 1,
        'top_k': top_k,
        'transformer_score_mean': float(np.mean(predictions['transformer'])),
        'transformer_score_std': float(np.std(predictions['transformer'])),
        'lgbm_score_mean': float(np.mean(predictions['lgbm'])),
        'lgbm_score_std': float(np.std(predictions['lgbm'])),
        'ensemble_score_mean': float(np.mean(predictions['ensemble'])),
        'ensemble_score_std': float(np.std(predictions['ensemble'])),
        'correlation_transformer_lgbm': float(np.corrcoef(predictions['transformer'], predictions['lgbm'])[0, 1])
    }
    
    return metrics, predictions


def main():
    """主函数：评估集成模型"""
    parser = argparse.ArgumentParser(description='评估集成模型')
    parser.add_argument('--config', type=str, default='config.py',
                       help='配置文件路径')
    parser.add_argument('--model-path', type=str,
                       help='集成模型路径')
    parser.add_argument('--output-dir', type=str, default='./evaluation',
                       help='输出目录')
    parser.add_argument('--top-k', type=int, default=5,
                       help='计算final_score时取前k个股票')
    parser.add_argument('--force', action='store_true',
                       help='强制重新评估，覆盖现有结果')
    args = parser.parse_args()
    
    print("=" * 60)
    print("集成模型评估")
    print("=" * 60)
    
    # 加载配置
    config = ENSEMBLE_CONFIG
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已有评估结果
    metrics_path = output_dir / 'evaluation_metrics.json'
    if metrics_path.exists() and not args.force:
        print(f"发现已有评估结果: {metrics_path}")
        print("使用 --force 参数强制重新评估")
        
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        print(f"\n现有评估结果:")
        print(f"  Transformer Final Score: {metrics['transformer_final_score']:.6f}")
        print(f"  LGBM Final Score:       {metrics['lgbm_final_score']:.6f}")
        print(f"  Ensemble Final Score:   {metrics['ensemble_final_score']:.6f}")
        print(f"  集成权重:               {metrics['weight']:.3f}")
        return
    
    # 加载模型
    print("\n1. 加载模型...")
    if args.model_path:
        try:
            ensemble_model = EnsembleModel.load(args.model_path, config)
            print(f"从指定路径加载模型: {args.model_path}")
        except Exception as e:
            print(f"从指定路径加载模型失败: {e}")
            print("尝试从配置加载模型...")
            ensemble_model = EnsembleModel.from_config(config)
    else:
        ensemble_model = EnsembleModel.from_config(config)
    
    # 显示模型信息
    model_info = ensemble_model.get_model_info()
    print(f"  集成权重: Transformer={model_info['transformer_weight']:.3f}, LGBM={model_info['lgbm_weight']:.3f}")
    
    # 准备验证数据
    print("\n2. 准备验证数据...")
    data_loader = EnsembleDataLoader(config)
    
    try:
        val_data = data_loader.prepare_validation_data()
    except Exception as e:
        print(f"准备验证数据失败: {e}")
        print("请确保数据文件存在且格式正确")
        raise
    
    if val_data['true_returns'] is None or len(val_data['true_returns']) == 0:
        raise ValueError("未能提取真实收益率，请检查数据")
    
    print(f"验证集股票数量: {len(val_data['true_returns'])}")
    print(f"真实收益率统计: 均值={np.mean(val_data['true_returns']):.4f}, 标准差={np.std(val_data['true_returns']):.4f}")
    
    # 评估模型
    print("\n3. 评估模型...")
    try:
        metrics, predictions = evaluate_ensemble(config, ensemble_model, val_data, args.top_k)
    except Exception as e:
        print(f"评估失败: {e}")
        raise
    
    # 保存评估结果
    print("\n4. 保存结果...")
    
    # 保存指标
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # 保存详细预测
    predictions_path = output_dir / 'detailed_predictions.csv'
    pred_df = pd.DataFrame({
        'stock_id': predictions['stock_ids'],
        'transformer_score': predictions['transformer'],
        'lgbm_score': predictions['lgbm'],
        'ensemble_score': predictions['ensemble'],
        'true_return': val_data['true_returns']
    })
    pred_df.to_csv(predictions_path, index=False)
    
    # 保存top-k分析
    top_k = args.top_k
    for model_name in ['transformer', 'lgbm', 'ensemble']:
        scores = predictions[model_name]
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:top_k]
        
        top_stocks = [predictions['stock_ids'][i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        top_returns = [val_data['true_returns'][i] for i in top_indices]
        
        top_df = pd.DataFrame({
            '排名': range(1, top_k + 1),
            '股票代码': top_stocks,
            '预测分数': top_scores,
            '真实收益': top_returns
        })
        
        top_path = output_dir / f'top{top_k}_{model_name}.csv'
        top_df.to_csv(top_path, index=False)
    
    # 打印结果
    print("\n评估结果:")
    print("-" * 60)
    print(f"Transformer Final Score: {metrics['transformer_final_score']:.6f}")
    print(f"LGBM Final Score:       {metrics['lgbm_final_score']:.6f}")
    print(f"Ensemble Final Score:   {metrics['ensemble_final_score']:.6f}")
    print(f"集成权重:               {metrics['weight']:.3f}")
    print(f"相比Transformer提升:    {metrics['improvement_vs_transformer']:.6f} ({metrics['improvement_percent_vs_transformer']:.1f}%)")
    print(f"相比LGBM提升:          {metrics['improvement_vs_lgbm']:.6f} ({metrics['improvement_percent_vs_lgbm']:.1f}%)")
    print(f"股票数量:              {metrics['num_stocks']}")
    print(f"Transformer-LGBM相关性: {metrics['correlation_transformer_lgbm']:.4f}")
    print("-" * 60)
    
    # 性能总结
    print("\n性能总结:")
    best_model = max(['transformer', 'lgbm', 'ensemble'], 
                     key=lambda x: metrics[f'{x}_final_score'])
    best_score = metrics[f'{best_model}_final_score']
    
    if best_model == 'ensemble':
        print(f"✓ 集成模型表现最佳，final_score = {best_score:.6f}")
        print(f"  相比最佳单一模型提升: {metrics['improvement_vs_transformer'] if metrics['transformer_final_score'] > metrics['lgbm_final_score'] else metrics['improvement_vs_lgbm']:.6f}")
    elif best_model == 'transformer':
        print(f"⚠ Transformer模型表现最佳，final_score = {best_score:.6f}")
        print(f"  集成模型未超越最佳单一模型，建议调整权重或重新训练")
    else:
        print(f"⚠ LGBM模型表现最佳，final_score = {best_score:.6f}")
        print(f"  集成模型未超越最佳单一模型，建议调整权重或重新训练")
    
    print(f"\n评估完成!")
    print(f"指标保存到: {metrics_path}")
    print(f"详细预测保存到: {predictions_path}")
    
    # 建议
    print("\n建议:")
    if metrics['improvement_vs_transformer'] > 0 and metrics['improvement_vs_lgbm'] > 0:
        print("✓ 集成模型在两个单一模型上都有提升，集成策略有效")
    elif metrics['improvement_vs_transformer'] > 0:
        print("✓ 集成模型相比Transformer有提升，但未超越LGBM")
        print("  考虑降低Transformer权重或优化LGBM模型")
    elif metrics['improvement_vs_lgbm'] > 0:
        print("✓ 集成模型相比LGBM有提升，但未超越Transformer")
        print("  考虑降低LGBM权重或优化Transformer模型")
    else:
        print("⚠ 集成模型未超越任一单一模型")
        print("  建议:")
        print("  1. 重新优化集成权重")
        print("  2. 检查模型预测是否对齐")
        print("  3. 考虑使用其他集成策略（如堆叠）")
    
    # 权重调整建议
    current_weight = metrics['weight']
    if metrics['transformer_final_score'] > metrics['lgbm_final_score']:
        suggestion = f"当前Transformer权重为{current_weight:.3f}，由于Transformer表现更好，可考虑增加权重"
    else:
        suggestion = f"当前Transformer权重为{current_weight:.3f}，由于LGBM表现更好，可考虑降低权重"
    
    print(f"  权重调整: {suggestion}")


if __name__ == '__main__':
    main()