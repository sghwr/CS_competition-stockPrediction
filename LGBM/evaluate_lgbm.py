"""
LGBM模型评估
评估已训练的LGBM模型在验证集上的性能
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 添加LGBM目录到sys.path，确保优先导入LGBM/config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "code", "src"))

# 导入LGBM配置和数据加载器
import lgbm_config
from data_loader import load_and_prepare_data, create_flat_dataset_for_lgbm
from train_lgbm import split_data_by_date, evaluate_lgbm_predictions

# 自定义 Numpy 编码器，解决 int64/float64 不能存 JSON 的问题
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def evaluate_lgbm_model(model_dir=None):
    """
    评估LGBM模型
    
    Args:
        model_dir: 模型目录，如果为None则使用config中的output_dir
    
    Returns:
        评估结果字典
    """
    print("=" * 60)
    print("LGBM模型评估")
    print("=" * 60)
    
    if model_dir is None:
        model_dir = r"E:\USELESS\数据分析学习\数分竞赛学习\CS-competition\LGBM\model"
    
    # 1. 加载模型和配置
    print("\n1. 加载模型和配置...")
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    model_path = os.path.join(model_dir, 'lgbm_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = joblib.load(model_path)
    print(f"模型已加载: {model_path}")
    
    # 2. 加载数据
    print("\n2. 加载数据...")
    data_dict = load_and_prepare_data(config)
    
    # 3. 创建扁平化数据集
    print("\n3. 创建扁平化数据集...")
    X, y, groups, dates, stock_ids = create_flat_dataset_for_lgbm(
        data_dict['train_processed'],
        data_dict['feature_columns'],
        config['sequence_length'],
        market_normalizer=None
    )
    
    # 4. 按日期划分验证集
    print("\n4. 划分验证集...")
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
    
    print(f"验证集样本数: {len(X_val)}")
    print(f"验证集日期数: {len(val_groups)}")
    
    # 5. 在验证集上预测
    print("\n5. 在验证集上预测...")
    y_pred_val = model.predict(X_val)
    
    # 6. 评估预测结果
    print("\n6. 评估预测结果...")
    results = evaluate_lgbm_predictions(
        y_pred_val, y_val, val_groups, val_dates, 
        stock_ids[dates >= val_start], k=5
    )
    
    # 7. 保存评估结果
    print("\n7. 保存评估结果...")
    eval_path = os.path.join(model_dir, 'evaluation_report.json')
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump({
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'validation_start_date': val_start.strftime('%Y-%m-%d'),
            'validation_end_date': last_date.strftime('%Y-%m-%d'),
            'num_validation_samples': len(X_val),
            'num_validation_dates': len(val_groups),
            'results': results
        }, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)  # 这里加了 cls=NumpyEncoder
    
    print(f"评估报告已保存: {eval_path}")
    
    # 8. 更新final_score
    final_score = results.get('final_score', 0.0)
    final_score_path = os.path.join(model_dir, 'final_score.txt')
    with open(final_score_path, 'w', encoding='utf-8') as f:
        f.write(f"Validation final_score: {final_score:.6f}\n")
        f.write(f"Evaluation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Final score已更新: {final_score_path}")
    
    # 9. 生成详细分析
    print("\n8. 生成详细分析...")
    generate_detailed_analysis(
        y_pred_val, y_val, val_groups, val_dates,
        stock_ids[dates >= val_start], model_dir
    )
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print(f"Final Score: {final_score:.6f}")
    print("=" * 60)
    
    return results


def generate_detailed_analysis(y_pred, y_true, groups, dates, stock_ids, output_dir):
    """
    生成详细分析报告
    
    Args:
        y_pred: 预测分数
        y_true: 真实标签
        groups: 分组信息
        dates: 日期向量
        stock_ids: 股票ID向量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 按日期分析
    unique_dates = np.unique(dates)
    date_results = []
    
    start_idx = 0
    for date_idx, date in enumerate(unique_dates):
        group_size = groups[date_idx]
        end_idx = start_idx + group_size
        
        if group_size < 5:
            start_idx = end_idx
            continue
        
        date_pred = y_pred[start_idx:end_idx]
        date_true = y_true[start_idx:end_idx]
        date_stocks = stock_ids[start_idx:end_idx]
        
        # 计算Top5命中率
        pred_top5 = np.argsort(date_pred)[::-1][:5]
        true_top5 = np.argsort(date_true)[::-1][:5]
        
        hit_rate = len(set(pred_top5) & set(true_top5)) / 5
        
        # 计算收益
        pred_return_sum = date_true[pred_top5].sum()
        max_return_sum = date_true[true_top5].sum()
        random_return_sum = 5 * date_true.mean()
        
        denominator = max_return_sum - random_return_sum
        if abs(denominator) > 1e-6:
            final_score = (pred_return_sum - random_return_sum) / denominator
        else:
            final_score = 0.0
        
        date_results.append({
            'date': str(date),
            'num_stocks': group_size,
            'hit_rate': hit_rate,
            'pred_return_sum': pred_return_sum,
            'max_return_sum': max_return_sum,
            'random_return_sum': random_return_sum,
            'final_score': final_score,
            'pred_top5_stocks': date_stocks[pred_top5].tolist(),
            'true_top5_stocks': date_stocks[true_top5].tolist(),
        })
        
        start_idx = end_idx
    
    # 保存日期级别结果
    if date_results:
        date_df = pd.DataFrame(date_results)
        date_csv_path = os.path.join(output_dir, 'datewise_analysis.csv')
        date_df.to_csv(date_csv_path, index=False, encoding='utf-8-sig')
        print(f"日期级别分析已保存: {date_csv_path}")
        
        # 计算统计摘要
        summary = {
            'avg_hit_rate': date_df['hit_rate'].mean(),
            'avg_final_score': date_df['final_score'].mean(),
            'positive_days': (date_df['final_score'] > 0).sum(),
            'negative_days': (date_df['final_score'] < 0).sum(),
            'best_day': date_df.loc[date_df['final_score'].idxmax()].to_dict() if len(date_df) > 0 else None,
            'worst_day': date_df.loc[date_df['final_score'].idxmin()].to_dict() if len(date_df) > 0 else None,
        }
        
        summary_path = os.path.join(output_dir, 'analysis_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"分析摘要已保存: {summary_path}")
        
        # 打印摘要
        print(f"\n分析摘要:")
        print(f"  平均命中率: {summary['avg_hit_rate']:.2%}")
        print(f"  平均Final Score: {summary['avg_final_score']:.6f}")
        print(f"  正收益天数: {summary['positive_days']}")
        print(f"  负收益天数: {summary['negative_days']}")
        
        if summary['best_day']:
            print(f"  最佳交易日: {summary['best_day']['date']}, Final Score: {summary['best_day']['final_score']:.6f}")
        
        if summary['worst_day']:
            print(f"  最差交易日: {summary['worst_day']['date']}, Final Score: {summary['worst_day']['final_score']:.6f}")


def main():
    """主函数"""
    try:
        results = evaluate_lgbm_model()
        
        # 判断模型效果
        final_score = results.get('final_score', 0.0)
        print(f"\n模型效果评估:")
        if final_score > 0.1:
            print("✅ 模型表现良好 (Final Score > 0.1)")
        elif final_score > 0.05:
            print("⚠️  模型表现一般 (0.05 < Final Score ≤ 0.1)")
        elif final_score > 0:
            print("⚠️  模型表现较弱 (0 < Final Score ≤ 0.05)")
        else:
            print("❌ 模型表现不佳 (Final Score ≤ 0)")
            
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()