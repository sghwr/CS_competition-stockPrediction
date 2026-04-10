"""
LightGBM回归模型训练脚本
目标：直接预测未来5日收益率，替代排序学习
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

from config import config
from utils import engineer_features_158plus39


def load_and_preprocess_data(data_path, sequence_length=60):
    """
    加载数据并进行特征工程，构建回归任务数据集
    每个样本：某只股票在某一天的特征 + 未来5日收益率标签
    """
    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])

    # 按股票代码分组进行特征工程
    print("进行特征工程...")
    groups = [group for _, group in df.groupby("股票代码", sort=False)]

    processed_list = []
    for group in groups:
        try:
            processed = engineer_features_158plus39(group)
            processed_list.append(processed)
        except Exception as e:
            print(f"特征工程失败: {group['股票代码'].iloc[0]}, 错误: {e}")
            continue

    if not processed_list:
        raise ValueError("特征工程后无有效数据")

    processed_data = pd.concat(processed_list).reset_index(drop=True)

    # 构建标签：未来5日收益率
    print("构建标签...")
    processed_data["open_t1"] = processed_data.groupby("股票代码")["开盘"].shift(-1)
    processed_data["open_t5"] = processed_data.groupby("股票代码")["开盘"].shift(-5)
    processed_data = processed_data[processed_data["open_t1"] > 1e-4]  # 过滤无效开盘价
    processed_data["label"] = (
        processed_data["open_t5"] - processed_data["open_t1"]
    ) / (processed_data["open_t1"] + 1e-12)
    processed_data = processed_data.dropna(subset=["label"])
    processed_data = processed_data.drop(columns=["open_t1", "open_t5"])

    # 确定特征列（排除非特征列）
    exclude_cols = ["股票代码", "日期", "label", "instrument"]
    feature_cols = [col for col in processed_data.columns if col not in exclude_cols]

    # 处理无穷值和NaN
    processed_data[feature_cols] = processed_data[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    # 删除包含NaN的行
    processed_data = processed_data.dropna(subset=feature_cols + ["label"])

    # 确保时序正确排序
    processed_data = processed_data.sort_values(["日期", "股票代码"]).reset_index(
        drop=True
    )

    print(f"数据集大小: {len(processed_data)} 行")
    print(f"特征数量: {len(feature_cols)}")
    print(
        f"日期范围: {processed_data['日期'].min().date()} 到 {processed_data['日期'].max().date()}"
    )

    return processed_data, feature_cols


def prepare_features_and_labels(data, feature_cols):
    """准备特征矩阵和标签向量"""
    X = data[feature_cols].values.astype(np.float32)
    y = data["label"].values.astype(np.float32)
    dates = data["日期"].values
    stock_ids = data["股票代码"].values

    return X, y, dates, stock_ids


def time_based_train_test_split(X, y, dates, test_size=0.2):
    """按时间划分训练集和测试集"""
    unique_dates = np.unique(dates)
    split_idx = int(len(unique_dates) * (1 - test_size))
    train_date_cutoff = unique_dates[split_idx]

    train_mask = dates < train_date_cutoff
    test_mask = dates >= train_date_cutoff

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    dates_train, dates_test = dates[train_mask], dates[test_mask]

    print(
        f"训练集: {len(X_train)} 样本, 日期范围: {dates_train.min()} 到 {dates_train.max()}"
    )
    print(
        f"测试集: {len(X_test)} 样本, 日期范围: {dates_test.min()} 到 {dates_test.max()}"
    )

    return X_train, X_test, y_train, y_test, dates_train, dates_test


def train_lightgbm_model(X_train, y_train, X_val, y_val, feature_names):
    """训练LightGBM回归模型"""
    print("\n训练LightGBM模型...")

    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(
        X_val, label=y_val, reference=train_data, feature_name=feature_names
    )

    # 参数设置
    params = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "max_depth": -1,  # 无限制
        "min_data_in_leaf": 20,
        "num_iterations": 1000,  # 增加迭代次数
        "early_stopping_round": 50,
    }

    # 训练模型
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(100)],  # 每100轮输出一次
    )

    return model


def evaluate_model(model, X_test, y_test, dates_test, stock_ids_test):
    """评估模型性能"""
    print("\n评估模型性能...")

    # 预测
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # 回归指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"测试集 MSE: {mse:.6f}")
    print(f"测试集 MAE: {mae:.6f}")
    print(f"测试集 RMSE: {np.sqrt(mse):.6f}")

    # 创建结果DataFrame
    results_df = pd.DataFrame(
        {
            "date": dates_test,
            "stock_id": stock_ids_test,
            "true_return": y_test,
            "pred_return": y_pred,
            "error": y_pred - y_test,
            "abs_error": np.abs(y_pred - y_test),
        }
    )

    # 按日期分析
    daily_stats = (
        results_df.groupby("date")
        .agg({"true_return": "mean", "pred_return": "mean", "abs_error": "mean"})
        .reset_index()
    )

    print(f"\n每日平均真实收益率: {daily_stats['true_return'].mean():.6f}")
    print(f"每日平均预测收益率: {daily_stats['pred_return'].mean():.6f}")
    print(f"每日平均绝对误差: {daily_stats['abs_error'].mean():.6f}")

    # 计算预测排名与真实排名的相关性（按日）
    daily_correlations = []
    for date, group in results_df.groupby("date"):
        if len(group) >= 10:  # 至少10只股票才计算相关性
            true_rank = group["true_return"].rank(method="average")
            pred_rank = group["pred_return"].rank(method="average")
            correlation = true_rank.corr(pred_rank)
            if not np.isnan(correlation):
                daily_correlations.append(correlation)

    if daily_correlations:
        avg_correlation = np.mean(daily_correlations)
        print(f"\n每日预测排名与真实排名平均相关性: {avg_correlation:.4f}")

    # 特征重要性
    importance_df = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 10最重要特征:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return results_df, importance_df


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

    output_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("LightGBM回归模型训练")
    print("=" * 60)
    print(f"数据文件: {data_path}")
    print(f"输出目录: {output_dir}")

    # 1. 加载和预处理数据
    processed_data, feature_cols = load_and_preprocess_data(
        data_path, sequence_length=config["sequence_length"]
    )

    # 2. 准备特征和标签
    X, y, dates, stock_ids = prepare_features_and_labels(processed_data, feature_cols)

    # 3. 按时间划分训练集和测试集（最后20%作为测试集）
    X_train, X_test, y_train, y_test, dates_train, dates_test = (
        time_based_train_test_split(X, y, dates, test_size=0.2)
    )

    # 4. 进一步划分训练集和验证集
    X_train_final, X_val, y_train_final, y_val, dates_train_final, dates_val = (
        time_based_train_test_split(X_train, y_train, dates_train, test_size=0.2)
    )

    # 5. 训练模型
    model = train_lightgbm_model(
        X_train_final, y_train_final, X_val, y_val, feature_cols
    )

    # 6. 评估模型
    stock_ids_test = stock_ids[dates == dates_test[0]][
        : len(X_test)
    ]  # 获取测试集股票ID
    results_df, importance_df = evaluate_model(
        model, X_test, y_test, dates_test, stock_ids_test
    )

    # 7. 保存模型和配置
    model_path = os.path.join(output_dir, "lgbm_model.pkl")
    feature_path = os.path.join(output_dir, "feature_columns.pkl")
    results_path = os.path.join(output_dir, "evaluation_results.csv")
    importance_path = os.path.join(output_dir, "feature_importance.csv")

    joblib.dump(model, model_path)
    joblib.dump(feature_cols, feature_path)
    results_df.to_csv(results_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    print(f"\n模型已保存: {model_path}")
    print(f"特征列已保存: {feature_path}")
    print(f"评估结果已保存: {results_path}")
    print(f"特征重要性已保存: {importance_path}")

    # 8. 输出模型性能摘要
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    # 计算Top5选择能力（在测试集上）
    print("\n测试集Top5选择能力分析:")
    test_results = results_df.copy()

    # 按日期分组分析
    top5_hit_rates = []
    for date, group in test_results.groupby("date"):
        if len(group) >= 20:  # 至少20只股票才分析
            true_top5 = group.nlargest(5, "true_return")["stock_id"].values
            pred_top5 = group.nlargest(5, "pred_return")["stock_id"].values
            hit_rate = len(set(true_top5) & set(pred_top5)) / 5
            top5_hit_rates.append(hit_rate)

    if top5_hit_rates:
        avg_hit_rate = np.mean(top5_hit_rates)
        print(f"平均Top5命中率: {avg_hit_rate:.2%}")

        # 计算预测Top5的平均真实收益率
        all_pred_top5 = test_results.nlargest(
            100, "pred_return"
        )  # 取预测最高的100个样本
        pred_top5_return = all_pred_top5["true_return"].mean()
        print(f"预测Top5平均真实收益率: {pred_top5_return:.4f}")

        # 计算真实Top5的平均收益率
        all_true_top5 = test_results.nlargest(100, "true_return")
        true_top5_return = all_true_top5["true_return"].mean()
        print(f"真实Top5平均收益率: {true_top5_return:.4f}")

        # 计算随机Top5的期望收益率
        random_returns = []
        for _ in range(1000):
            random_sample = test_results.sample(5, replace=False)
            random_returns.append(random_sample["true_return"].mean())
        random_return = np.mean(random_returns)
        print(f"随机策略平均收益率: {random_return:.4f}")

        print(f"模型超额收益: {pred_top5_return - random_return:.4f}")


if __name__ == "__main__":
    main()
