"""
StockTransformer分数与实际收益率相关性分析
分析模型排序分数与未来5日收益率的统计关系
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
from scipy import stats
from tqdm import tqdm

# 添加项目路径以便导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "src"))

from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39

# 设置matplotlib中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 特征工程映射
feature_columns_map = {
    "39": [
        "instrument",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌额",
        "换手率",
        "涨跌幅",
        "sma_5",
        "sma_20",
        "ema_12",
        "ema_26",
        "rsi",
        "macd",
        "macd_signal",
        "volume_change",
        "obv",
        "volume_ma_5",
        "volume_ma_20",
        "volume_ratio",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "boll_mid",
        "boll_std",
        "atr_14",
        "ema_60",
        "volatility_10",
        "volatility_20",
        "return_1",
        "return_5",
        "return_10",
        "high_low_spread",
        "open_close_spread",
        "high_close_spread",
        "low_close_spread",
    ],
    "158+39": [
        "instrument",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌额",
        "换手率",
        "涨跌幅",
        "KMID",
        "KLEN",
        "KMID2",
        "KUP",
        "KUP2",
        "KLOW",
        "KLOW2",
        "KSFT",
        "KSFT2",
        "OPEN0",
        "HIGH0",
        "LOW0",
        "VWAP0",
        "ROC5",
        "ROC10",
        "ROC20",
        "ROC30",
        "ROC60",
        "MA5",
        "MA10",
        "MA20",
        "MA30",
        "MA60",
        "STD5",
        "STD10",
        "STD20",
        "STD30",
        "STD60",
        "BETA5",
        "BETA10",
        "BETA20",
        "BETA30",
        "BETA60",
        "RSQR5",
        "RSQR10",
        "RSQR20",
        "RSQR30",
        "RSQR60",
        "RESI5",
        "RESI10",
        "RESI20",
        "RESI30",
        "RESI60",
        "MAX5",
        "MAX10",
        "MAX20",
        "MAX30",
        "MAX60",
        "MIN5",
        "MIN10",
        "MIN20",
        "MIN30",
        "MIN60",
        "QTLU5",
        "QTLU10",
        "QTLU20",
        "QTLU30",
        "QTLU60",
        "QTLD5",
        "QTLD10",
        "QTLD20",
        "QTLD30",
        "QTLD60",
        "RANK5",
        "RANK10",
        "RANK20",
        "RANK30",
        "RANK60",
        "RSV5",
        "RSV10",
        "RSV20",
        "RSV30",
        "RSV60",
        "IMAX5",
        "IMAX10",
        "IMAX20",
        "IMAX30",
        "IMAX60",
        "IMIN5",
        "IMIN10",
        "IMIN20",
        "IMIN30",
        "IMIN60",
        "IMXD5",
        "IMXD10",
        "IMXD20",
        "IMXD30",
        "IMXD60",
        "CORR5",
        "CORR10",
        "CORR20",
        "CORR30",
        "CORR60",
        "CORD5",
        "CORD10",
        "CORD20",
        "CORD30",
        "CORD60",
        "CNTP5",
        "CNTP10",
        "CNTP20",
        "CNTP30",
        "CNTP60",
        "CNTN5",
        "CNTN10",
        "CNTN20",
        "CNTN30",
        "CNTN60",
        "CNTD5",
        "CNTD10",
        "CNTD20",
        "CNTD30",
        "CNTD60",
        "SUMP5",
        "SUMP10",
        "SUMP20",
        "SUMP30",
        "SUMP60",
        "SUMN5",
        "SUMN10",
        "SUMN20",
        "SUMN30",
        "SUMN60",
        "SUMD5",
        "SUMD10",
        "SUMD20",
        "SUMD30",
        "SUMD60",
        "VMA5",
        "VMA10",
        "VMA20",
        "VMA30",
        "VMA60",
        "VSTD5",
        "VSTD10",
        "VSTD20",
        "VSTD30",
        "VSTD60",
        "WVMA5",
        "WVMA10",
        "WVMA20",
        "WVMA30",
        "WVMA60",
        "VSUMP5",
        "VSUMP10",
        "VSUMP20",
        "VSUMP30",
        "VSUMP60",
        "VSUMN5",
        "VSUMN10",
        "VSUMN20",
        "VSUMN30",
        "VSUMN60",
        "VSUMD5",
        "VSUMD10",
        "VSUMD20",
        "VSUMD30",
        "VSUMD60",
        "sma_5",
        "sma_20",
        "ema_12",
        "ema_26",
        "rsi",
        "macd",
        "macd_signal",
        "volume_change",
        "obv",
        "volume_ma_5",
        "volume_ma_20",
        "volume_ratio",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "boll_mid",
        "boll_std",
        "atr_14",
        "ema_60",
        "volatility_10",
        "volatility_20",
        "return_1",
        "return_5",
        "return_10",
        "high_low_spread",
        "open_close_spread",
        "high_close_spread",
        "low_close_spread",
    ],
}

feature_engineer_func_map = {
    "39": engineer_features_39,
    "158+39": engineer_features_158plus39,
}


def load_and_prepare_data(train_path, test_path, sequence_length, feature_num):
    """加载训练和测试数据，准备特征工程"""
    print(f"加载训练数据: {train_path}")
    train_df = pd.read_csv(train_path, dtype={"股票代码": str})
    train_df["股票代码"] = train_df["股票代码"].astype(str).str.zfill(6)
    train_df["日期"] = pd.to_datetime(train_df["日期"])

    print(f"加载测试数据: {test_path}")
    test_df = pd.read_csv(test_path, dtype={"股票代码": str})
    test_df["股票代码"] = test_df["股票代码"].astype(str).str.zfill(6)
    test_df["日期"] = pd.to_datetime(test_df["日期"])

    # 确定预测基准日T：train_df中的最新日期
    T_date = train_df["日期"].max()
    print(f"预测基准日 T: {T_date.date()}")

    # 测试数据的日期范围
    test_dates = sorted(test_df["日期"].unique())
    print(f"测试数据日期范围: {test_dates[0].date()} 到 {test_dates[-1].date()}")
    print(f"测试数据包含 {len(test_dates)} 个交易日")

    # 获取所有股票ID
    all_stock_ids = sorted(train_df["股票代码"].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}
    num_stocks = len(stockid2idx)
    print(f"股票总数: {num_stocks}")

    # 特征工程函数
    assert feature_num in feature_engineer_func_map, f"不支持的特征数: {feature_num}"
    feature_engineer = feature_engineer_func_map[feature_num]
    feature_columns = feature_columns_map[feature_num]

    # 为预测准备数据：使用截至T日的数据
    print("准备预测数据...")
    pred_df = train_df[train_df["日期"] <= T_date].copy()

    # 按股票分组进行特征工程
    groups = [group for _, group in pred_df.groupby("股票代码", sort=False)]

    # 使用单进程处理（简化）
    processed_list = []
    for group in tqdm(groups, desc="特征工程"):
        processed = feature_engineer(group)
        processed_list.append(processed)

    processed_data = pd.concat(processed_list).reset_index(drop=True)
    processed_data["instrument"] = processed_data["股票代码"].map(stockid2idx)
    processed_data = processed_data.dropna(subset=["instrument"]).copy()
    processed_data["instrument"] = processed_data["instrument"].astype(np.int64)

    # 处理无穷值和NaN
    processed_data[feature_columns] = (
        processed_data[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    return {
        "train_df": train_df,
        "test_df": test_df,
        "T_date": T_date,
        "stockid2idx": stockid2idx,
        "num_stocks": num_stocks,
        "processed_data": processed_data,
        "feature_columns": feature_columns,
        "all_stock_ids": all_stock_ids,
    }


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
    """构建推理序列"""
    sequences, sequence_stock_ids = [], []
    for stock_id in stock_ids:
        stock_history = (
            data[(data["股票代码"] == stock_id) & (data["日期"] <= latest_date)]
            .sort_values("日期")
            .tail(sequence_length)
        )

        if len(stock_history) == sequence_length:
            sequences.append(stock_history[features].values.astype(np.float32))
            sequence_stock_ids.append(stock_id)

    if len(sequences) == 0:
        raise ValueError("没有可用于预测的股票序列")

    return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def calculate_real_returns(test_df, stock_ids, T_date):
    """计算真实5日收益率"""
    # 确定T+1和T+5日期
    test_dates = sorted(test_df["日期"].unique())
    if len(test_dates) < 5:
        raise ValueError(f"测试数据不足5个交易日: {len(test_dates)}")

    # 假设test_df包含T+1到T+5的数据
    T_plus_1 = test_dates[0]
    T_plus_5 = test_dates[4]

    print(f"T+1日期: {T_plus_1.date()}, T+5日期: {T_plus_5.date()}")

    returns = {}
    for stock_id in stock_ids:
        stock_data = test_df[test_df["股票代码"] == stock_id].sort_values("日期")

        # 获取T+1和T+5的开盘价
        open_t1 = stock_data[stock_data["日期"] == T_plus_1]["开盘"]
        open_t5 = stock_data[stock_data["日期"] == T_plus_5]["开盘"]

        if len(open_t1) > 0 and len(open_t5) > 0:
            return_val = (open_t5.values[0] - open_t1.values[0]) / open_t1.values[0]
            returns[stock_id] = return_val
        else:
            returns[stock_id] = np.nan

    return returns, T_plus_1, T_plus_5


def analyze_correlation(scores_dict, returns_dict):
    """分析分数与收益率的相关性"""
    # 对齐数据
    common_stocks = list(set(scores_dict.keys()) & set(returns_dict.keys()))

    scores = []
    returns = []
    valid_stocks = []

    for stock in common_stocks:
        if not np.isnan(returns_dict[stock]):
            scores.append(scores_dict[stock])
            returns.append(returns_dict[stock])
            valid_stocks.append(stock)

    scores = np.array(scores)
    returns = np.array(returns)

    print(f"有效股票数量: {len(valid_stocks)}")

    if len(scores) < 10:
        print("有效数据不足，无法进行相关性分析")
        return None

    # 1. Spearman秩相关系数
    spearman_corr, spearman_p = stats.spearmanr(scores, returns)

    # 2. Pearson相关系数
    pearson_corr, pearson_p = stats.pearsonr(scores, returns)

    # 3. Top5命中率
    sorted_by_score = np.argsort(scores)[::-1]  # 按分数降序
    sorted_by_return = np.argsort(returns)[::-1]  # 按收益率降序

    top5_by_score = set(valid_stocks[i] for i in sorted_by_score[:5])
    top5_by_return = set(valid_stocks[i] for i in sorted_by_return[:5])
    top5_hit_rate = len(top5_by_score & top5_by_return) / 5

    # 4. 分数分档收益分析
    n_bins = 5
    score_bins = pd.qcut(scores, n_bins, labels=False, duplicates="drop")

    bin_stats = []
    for bin_idx in range(n_bins):
        mask = score_bins == bin_idx
        if mask.sum() > 0:
            bin_mean_score = scores[mask].mean()
            bin_mean_return = returns[mask].mean()
            bin_std_return = returns[mask].std()
            bin_size = mask.sum()
            bin_stats.append(
                {
                    "bin": bin_idx,
                    "mean_score": bin_mean_score,
                    "mean_return": bin_mean_return,
                    "std_return": bin_std_return,
                    "size": bin_size,
                }
            )

    results = {
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p,
        "top5_hit_rate": top5_hit_rate,
        "top5_score_stocks": list(top5_by_score),
        "top5_return_stocks": list(top5_by_return),
        "bin_stats": bin_stats,
        "scores": scores,
        "returns": returns,
        "stocks": valid_stocks,
        "score_dict": scores_dict,
        "return_dict": returns_dict,
    }

    return results


def plot_correlation_results(results, output_dir):
    """绘制相关性分析图表"""
    os.makedirs(output_dir, exist_ok=True)

    scores = results["scores"]
    returns = results["returns"]
    stocks = results["stocks"]

    # 1. 散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(scores, returns, alpha=0.6, s=30)
    plt.xlabel("模型分数 (Score)")
    plt.ylabel("5日真实收益率 (Return)")
    plt.title(
        f"分数 vs 收益率 (Spearman={results['spearman_corr']:.3f}, Pearson={results['pearson_corr']:.3f})"
    )
    plt.grid(True, alpha=0.3)

    # 添加回归线
    if len(scores) > 1:
        z = np.polyfit(scores, returns, 1)
        p = np.poly1d(z)
        plt.plot(sorted(scores), p(sorted(scores)), "r--", alpha=0.8, linewidth=2)

    scatter_path = os.path.join(output_dir, "score_vs_return_scatter.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"散点图已保存: {scatter_path}")

    # 2. 排名对比图
    plt.figure(figsize=(12, 6))

    # 按分数排名
    score_rank = np.argsort(scores)[::-1]  # 降序
    return_values_by_score_rank = returns[score_rank]

    # 按收益率排名
    return_rank = np.argsort(returns)[::-1]
    score_values_by_return_rank = scores[return_rank]

    plt.subplot(1, 2, 1)
    plt.plot(
        range(len(return_values_by_score_rank)),
        return_values_by_score_rank,
        "b-",
        alpha=0.7,
        linewidth=1,
    )
    plt.scatter(range(5), return_values_by_score_rank[:5], color="red", s=50, zorder=5)
    plt.xlabel("按分数排名 (Rank by Score)")
    plt.ylabel("收益率 (Return)")
    plt.title("按分数排名的收益率分布")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(
        range(len(score_values_by_return_rank)),
        score_values_by_return_rank,
        "g-",
        alpha=0.7,
        linewidth=1,
    )
    plt.scatter(range(5), score_values_by_return_rank[:5], color="red", s=50, zorder=5)
    plt.xlabel("按收益率排名 (Rank by Return)")
    plt.ylabel("分数 (Score)")
    plt.title("按收益率排名的分数分布")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    rank_path = os.path.join(output_dir, "ranking_comparison.png")
    plt.savefig(rank_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"排名对比图已保存: {rank_path}")

    # 3. 分档收益图
    bin_stats = results["bin_stats"]
    if bin_stats:
        plt.figure(figsize=(10, 6))

        bins = [s["bin"] for s in bin_stats]
        mean_returns = [s["mean_return"] for s in bin_stats]
        std_returns = [s["std_return"] for s in bin_stats]

        plt.bar(bins, mean_returns, yerr=std_returns, capsize=5, alpha=0.7)
        plt.xlabel("分数分档 (Score Bins)")
        plt.ylabel("平均收益率 (Mean Return)")
        plt.title("分数分档的平均收益率（误差棒为标准差）")
        plt.grid(True, alpha=0.3, axis="y")

        # 在柱子上方标注数量
        for i, stats in enumerate(bin_stats):
            plt.text(
                stats["bin"],
                stats["mean_return"] + 0.001,
                f"n={stats['size']}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        bin_path = os.path.join(output_dir, "bin_return_analysis.png")
        plt.savefig(bin_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"分档收益图已保存: {bin_path}")

    # 4. Top5对比表格
    top5_score = results["top5_score_stocks"]
    top5_return = results["top5_return_stocks"]

    # 创建对比DataFrame
    comparison_data = []
    for i in range(5):
        if i < len(top5_score):
            score_stock = top5_score[i]
            score_rank = i + 1
            score_return = results["return_dict"].get(score_stock, np.nan)
        else:
            score_stock = "-"
            score_rank = "-"
            score_return = np.nan

        if i < len(top5_return):
            return_stock = top5_return[i]
            return_rank = i + 1
            return_score = results["score_dict"].get(return_stock, np.nan)
        else:
            return_stock = "-"
            return_rank = "-"
            return_score = np.nan

        comparison_data.append(
            {
                "模型Top5排名": score_rank,
                "股票代码": score_stock,
                "实际收益率": f"{score_return:.4f}"
                if not np.isnan(score_return)
                else "-",
                "真实Top5排名": return_rank,
                "股票代码": return_stock,
                "模型分数": f"{return_score:.4f}"
                if not np.isnan(return_score)
                else "-",
            }
        )

    df_comparison = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, "top5_comparison.csv")
    df_comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    print(f"Top5对比表格已保存: {comparison_path}")

    return {
        "scatter_plot": scatter_path,
        "rank_plot": rank_path,
        "bin_plot": bin_path if bin_stats else None,
        "comparison_csv": comparison_path,
    }


def main():
    """主函数"""
    # 配置文件 - 使用相对于脚本的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # 上一级目录是项目根目录

    train_path = os.path.join(project_root, "data", "train.csv")
    test_path = os.path.join(project_root, "data", "test.csv")
    model_dir = os.path.join(project_root, "model", "60_158+39")
    output_dir = os.path.join(script_dir, "correlation_results")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("StockTransformer分数与收益率相关性分析")
    print("=" * 60)

    # 1. 加载和准备数据
    data_dict = load_and_prepare_data(
        train_path, test_path, config["sequence_length"], config["feature_num"]
    )

    # 2. 构建推理序列
    print("\n构建推理序列...")
    sequences_np, sequence_stock_ids = build_inference_sequences(
        data_dict["processed_data"],
        data_dict["feature_columns"],
        config["sequence_length"],
        data_dict["all_stock_ids"],
        data_dict["T_date"],
    )
    print(f"构建了 {len(sequence_stock_ids)} 只股票的序列")

    # 3. 加载模型和scaler
    print("\n加载模型和标准化器...")
    model_path = os.path.join(model_dir, "best_model.pth")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")

    scaler = joblib.load(scaler_path)

    # 标准化特征
    feature_columns = data_dict["feature_columns"]
    data_dict["processed_data"][feature_columns] = scaler.transform(
        data_dict["processed_data"][feature_columns]
    )

    # 重新构建序列（标准化后）
    sequences_np, sequence_stock_ids = build_inference_sequences(
        data_dict["processed_data"],
        feature_columns,
        config["sequence_length"],
        data_dict["all_stock_ids"],
        data_dict["T_date"],
    )

    # 4. 模型推理
    print("\n运行模型推理...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    model = StockTransformer(
        input_dim=len(feature_columns),
        config=config,
        num_stocks=len(data_dict["all_stock_ids"]),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
        scores = model(x).squeeze(0).detach().cpu().numpy()  # [N]

    # 创建分数字典
    scores_dict = {
        stock_id: score for stock_id, score in zip(sequence_stock_ids, scores)
    }
    print(f"模型推理完成，得到 {len(scores_dict)} 只股票的分数")

    # 5. 计算真实收益率
    print("\n计算真实收益率...")
    returns_dict, T_plus_1, T_plus_5 = calculate_real_returns(
        data_dict["test_df"], sequence_stock_ids, data_dict["T_date"]
    )
    print(
        f"计算了 {sum(1 for v in returns_dict.values() if not np.isnan(v))} 只股票的真实收益率"
    )

    # 6. 相关性分析
    print("\n进行相关性分析...")
    results = analyze_correlation(scores_dict, returns_dict)

    if results is None:
        print("相关性分析失败，数据不足")
        return

    # 7. 打印分析结果
    print("\n" + "=" * 60)
    print("相关性分析结果")
    print("=" * 60)
    print(
        f"Spearman秩相关系数: {results['spearman_corr']:.4f} (p={results['spearman_p']:.4f})"
    )
    print(
        f"Pearson相关系数: {results['pearson_corr']:.4f} (p={results['pearson_p']:.4f})"
    )
    print(f"Top5命中率: {results['top5_hit_rate']:.2%}")

    # 解释相关性强度
    spearman_abs = abs(results["spearman_corr"])
    if spearman_abs >= 0.7:
        strength = "强"
    elif spearman_abs >= 0.4:
        strength = "中等"
    elif spearman_abs >= 0.2:
        strength = "弱"
    else:
        strength = "非常弱或无"
    print(f"相关性强度: {strength}")

    # 打印分档统计
    print("\n分数分档收益率分析:")
    for stats in results["bin_stats"]:
        print(
            f"  分档{stats['bin'] + 1}: 平均分数={stats['mean_score']:.4f}, "
            f"平均收益率={stats['mean_return']:.4f}, "
            f"样本数={stats['size']}"
        )

    # 8. 可视化
    print("\n生成可视化图表...")
    plot_paths = plot_correlation_results(results, output_dir)

    # 9. 保存完整结果
    summary = {
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "T_date": data_dict["T_date"].strftime("%Y-%m-%d"),
        "T_plus_1": T_plus_1.strftime("%Y-%m-%d"),
        "T_plus_5": T_plus_5.strftime("%Y-%m-%d"),
        "num_stocks_total": len(data_dict["all_stock_ids"]),
        "num_stocks_with_score": len(scores_dict),
        "num_stocks_with_return": sum(
            1 for v in returns_dict.values() if not np.isnan(v)
        ),
        "spearman_correlation": float(results["spearman_corr"]),
        "spearman_p_value": float(results["spearman_p"]),
        "pearson_correlation": float(results["pearson_corr"]),
        "pearson_p_value": float(results["pearson_p"]),
        "top5_hit_rate": float(results["top5_hit_rate"]),
        "top5_score_stocks": results["top5_score_stocks"],
        "top5_return_stocks": results["top5_return_stocks"],
        "plot_paths": plot_paths,
    }

    summary_path = os.path.join(output_dir, "correlation_summary.json")
    import json

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"分析摘要已保存: {summary_path}")

    # 10. 保存原始数据
    raw_data = pd.DataFrame(
        {
            "stock_id": results["stocks"],
            "score": results["scores"],
            "return": results["returns"],
        }
    ).sort_values("score", ascending=False)

    raw_data_path = os.path.join(output_dir, "score_return_data.csv")
    raw_data.to_csv(raw_data_path, index=False, encoding="utf-8-sig")
    print(f"原始数据已保存: {raw_data_path}")

    print("\n" + "=" * 60)
    print("分析完成!")
    print(f"所有结果已保存到: {output_dir}")
    print("=" * 60)

    # 提供结论和建议
    print("\n结论与建议:")
    if abs(results["spearman_corr"]) >= 0.3:
        print("✅ 模型分数与实际收益率有显著相关性，支持基于分数的权重优化。")
        print("   建议: 使用softmax函数将分数转换为权重，可引入温度参数调节权重分布。")
    elif abs(results["spearman_corr"]) >= 0.15:
        print("⚠️  模型分数与实际收益率有弱相关性，权重优化可能效果有限。")
        print("   建议: 可尝试权重优化，但需结合其他信号或采用更保守的策略。")
    else:
        print("❌ 模型分数与实际收益率相关性很弱，直接基于分数的权重优化可能无效。")
        print("   建议: 考虑重新设计模型输出层或特征工程，或采用等权重策略。")

    if results["top5_hit_rate"] >= 0.4:
        print(f"✅ Top5命中率较高 ({results['top5_hit_rate']:.0%})，模型选股能力良好。")
    else:
        print(
            f"⚠️  Top5命中率较低 ({results['top5_hit_rate']:.0%})，模型选股准确性有待提升。"
        )


if __name__ == "__main__":
    main()
