"""
LGBM模型评估与评分集成脚本
一键完成预测、评分、结果记录
"""
import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_prediction(team_name='lgbm_model'):
    """
    运行LGBM预测，生成评分兼容格式的结果
    
    Args:
        team_name: 团队名称，用于评分
    """
    print("=" * 60)
    print("步骤1: 运行LGBM预测")
    print("=" * 60)
    
    # 构建命令 - 在LGBM目录中运行
    lgbm_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "predict_lgbm.py", "--score-format", "--team-name", team_name]
    
    print(f"执行命令: {' '.join(cmd)}")
    print(f"工作目录: {lgbm_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=lgbm_dir)
    
    if result.returncode != 0:
        print(f"预测失败: {result.stderr}")
        raise RuntimeError(f"预测过程出错: {result.returncode}")
    
    print(result.stdout)
    
    # 检查输出文件
    output_path = os.path.join(project_root, 'output', 'result.csv')
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"预测输出文件不存在: {output_path}")
    
    print(f"预测结果已生成: {output_path}")
    return output_path

def run_scoring(team_name='lgbm_model'):
    """
    运行评分脚本
    
    Args:
        team_name: 团队名称
    """
    print("\n" + "=" * 60)
    print("步骤2: 运行评分")
    print("=" * 60)
    
    # 构建命令
    score_script = os.path.join(project_root, 'test', 'score_docker.py')
    cmd = [sys.executable, score_script, team_name]
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"评分失败: {result.stderr}")
        raise RuntimeError(f"评分过程出错: {result.returncode}")
    
    print(result.stdout)
    
    # 检查临时结果文件
    tmp_result_path = os.path.join(project_root, 'temp', 'tmp.csv')
    if not os.path.exists(tmp_result_path):
        raise FileNotFoundError(f"临时评分结果文件不存在: {tmp_result_path}")
    
    print(f"评分完成，临时结果: {tmp_result_path}")
    return tmp_result_path

def merge_results(team_name='lgbm_model'):
    """
    合并评分结果到总结果文件
    
    Args:
        team_name: 团队名称
    """
    print("\n" + "=" * 60)
    print("步骤3: 合并结果")
    print("=" * 60)
    
    # 读取临时结果
    tmp_path = os.path.join(project_root, 'temp', 'tmp.csv')
    df1 = pd.read_csv(tmp_path)
    print(f"临时结果:\n{df1}")
    
    # 读取或创建总结果文件
    result_path = os.path.join(project_root, 'test', 'result.csv')
    try:
        df2 = pd.read_csv(result_path)
        print(f"现有结果文件: {result_path} ({len(df2)} 条记录)")
    except FileNotFoundError:
        print(f"结果文件不存在，创建新文件: {result_path}")
        df2 = pd.DataFrame(columns=["Team Name", "Final Score"])
    
    # 合并结果
    df_combined = pd.concat([df2, df1], ignore_index=True)
    
    # 保存合并后的结果
    df_combined.to_csv(result_path, index=False)
    print(f"结果已合并到: {result_path} (总计 {len(df_combined)} 条记录)")
    
    # 显示最终结果
    print("\n最终结果汇总:")
    print("-" * 40)
    print(df_combined.tail(10))  # 显示最后10条记录
    print("-" * 40)
    
    return result_path

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LGBM模型评估与评分集成脚本')
    parser.add_argument('--team-name', type=str, default='lgbm_model',
                       help='团队名称，用于评分（默认: lgbm_model）')
    parser.add_argument('--skip-prediction', action='store_true',
                       help='跳过预测步骤，直接使用现有结果文件')
    parser.add_argument('--skip-scoring', action='store_true',
                       help='跳过评分步骤，只合并现有结果')
    args = parser.parse_args()
    
    print("LGBM模型评估与评分集成脚本")
    print(f"团队名称: {args.team_name}")
    print(f"项目根目录: {project_root}")
    print()
    
    try:
        # 步骤1: 运行预测（除非跳过）
        if not args.skip_prediction:
            run_prediction(args.team_name)
        else:
            print("跳过预测步骤")
        
        # 步骤2: 运行评分（除非跳过）
        if not args.skip_scoring:
            run_scoring(args.team_name)
        else:
            print("跳过评分步骤")
        
        # 步骤3: 合并结果
        result_path = merge_results(args.team_name)
        
        print("\n" + "=" * 60)
        print("评估与评分完成!")
        print("=" * 60)
        print(f"最终结果文件: {result_path}")
        
        # 显示最终分数
        final_df = pd.read_csv(result_path)
        latest_score = final_df.iloc[-1]['Final Score']
        print(f"最新分数 (团队: {args.team_name}): {latest_score:.6f}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()