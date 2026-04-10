"""
LightGBM模型预测脚本
基于训练好的模型预测未来收益率，选择Top5股票
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# 添加项目路径以便导入模块
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "code", "src"
    ),
)

from utils import engineer_features_158plus39


def load_model_and_features(model_dir):
    """加载训练好的模型和特征列"""
    model_path = os.path.join(model_dir, "lgbm_model.pkl")
    feature_path = os.path.join(model_dir, "feature_columns.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"特征列文件不存在: {feature_path}")

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)

    print(f"模型加载成功，特征数量: {len(feature_cols)}")
    print(f"模型最佳迭代次数: {model.best_iteration}")

    return model, feature_cols


def prepare_prediction_data(data_path, feature_cols):
    """
    准备预测数据
    返回最新交易日每只股票的特征
    """
    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])

    # 获取最新交易日
    latest_date = df["日期"].max()
    print(f"最新交易日: {latest_date.date()}")

    # 按股票代码分组进行特征工程
    print("进行特征工程...")
    groups = [group for _, group in df.groupby("股票代码", sort=False)]

    prediction_data_list = []
    stock_ids = []

    for group in groups:
        stock_id = group["股票代码"].iloc[0]
        try:
            # 对每只股票进行特征工程
            processed = engineer_features_158plus39(group)

            # 获取最新交易日的特征
            latest_row = processed[processed["日期"] == latest_date]
            if len(latest_row) > 0:
                # 检查是否包含所有需要的特征列
                row = latest_row.iloc[0]
                missing_features = [f for f in feature_cols if f not in row.index]
                if missing_features:
                    print(f"警告: 股票 {stock_id} 缺少特征: {missing_features[:5]}...")
                    continue

                # 提取特征值
                features = row[feature_cols].values.astype(np.float32)
                prediction_data_list.append(features)
                stock_ids.append(stock_id)

        except Exception as e:
            print(f"股票 {stock_id} 特征工程失败: {e}")
            continue

    if not prediction_data_list:
        raise ValueError("无有效预测数据")

    X_pred = np.array(prediction_data_list)

    print(f"可用于预测的股票数量: {len(stock_ids)}")
    return X_pred, stock_ids, latest_date


def make_predictions(model, X_pred, stock_ids):
    """使用模型进行预测"""
    print("\n进行收益率预测...")

    # 预测
    y_pred = model.predict(X_pred, num_iteration=model.best_iteration)

    # 创建预测结果DataFrame
    predictions_df = pd.DataFrame({"stock_id": stock_ids, "pred_return": y_pred})

    # 按预测收益率降序排序
    predictions_df = predictions_df.sort_values(
        "pred_return", ascending=False
    ).reset_index(drop=True)

    print(f"预测收益率范围: {y_pred.min():.4f} 到 {y_pred.max():.4f}")
    print(f"预测收益率均值: {y_pred.mean():.4f}")
    print(f"预测收益率标准差: {y_pred.std():.4f}")

    # 显示Top10预测结果
    print("\n预测收益率Top10股票:")
    for i, row in predictions_df.head(10).iterrows():
        print(f"  {i + 1}. {row['stock_id']}: {row['pred_return']:.4f}")

    return predictions_df


def select_top5_stocks(predictions_df):
    """选择预测收益率最高的5只股票"""
    if len(predictions_df) < 5:
        raise ValueError(f"可预测股票不足5只，当前仅有 {len(predictions_df)} 只")

    top5 = predictions_df.head(5).copy()

    print("\n选定的Top5股票:")
    for i, row in top5.iterrows():
        print(f"  {i + 1}. {row['stock_id']}: 预测收益率={row['pred_return']:.4f}")

    return top5


def save_results(top5_df, output_path, latest_date):
    """保存预测结果到CSV文件"""
    # 创建输出DataFrame，使用等权重0.2
    output_df = pd.DataFrame(
        {"stock_id": top5_df["stock_id"].values, "weight": [0.2] * len(top5_df)}
    )

    # 保存到CSV
    output_df.to_csv(output_path, index=False)

    print(f"\n预测日期: {latest_date.date()}")
    print(f"结果已保存: {output_path}")
    print("\n结果预览:")
    print(output_df.to_string(index=False))

    # 计算总权重
    total_weight = output_df["weight"].sum()
    print(f"总权重: {total_weight:.2f} (剩余现金权重: {1 - total_weight:.2f})")


def main(data_path=None):
    """主函数
    Args:
        data_path: 数据文件路径，如果为None则尝试从环境变量或默认位置获取
    """
    # 配置文件
    if data_path is None:
        # 尝试从环境变量获取
        data_path = os.environ.get("DATA_PATH")
        if data_path is None:
            # 使用默认路径
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data",
                "train.csv",
            )

    model_dir = os.path.join(os.path.dirname(__file__), "model")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    output_path = os.path.join(output_dir, "result.csv")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("LightGBM模型预测")
    print("=" * 60)
    print(f"数据文件: {data_path}")
    print(f"模型目录: {model_dir}")
    print(f"输出目录: {output_dir}")

    try:
        # 1. 加载模型和特征列
        model, feature_cols = load_model_and_features(model_dir)

        # 2. 准备预测数据
        X_pred, stock_ids, latest_date = prepare_prediction_data(
            data_path, feature_cols
        )

        # 3. 进行预测
        predictions_df = make_predictions(model, X_pred, stock_ids)

        # 4. 选择Top5股票
        top5_df = select_top5_stocks(predictions_df)

        # 5. 保存结果
        save_results(top5_df, output_path, latest_date)

        # 6. 保存完整的预测结果供分析
        full_predictions_path = os.path.join(output_dir, "full_predictions.csv")
        predictions_df.to_csv(full_predictions_path, index=False)
        print(f"\n完整预测结果已保存: {full_predictions_path}")

    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("预测完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
