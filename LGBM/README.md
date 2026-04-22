# LGBM股票排序模型

基于LightGBM LambdaRank的股票排序模型，复用现有Transformer代码库的数据预处理、特征工程和评估指标。

## 核心特性

- **修正标签计算**: 使用 `(open_{T+5} - open_{T+1}) / open_{T+1}` 收益率，与竞赛实际交易场景一致
- **排序学习**: 使用LightGBM的LambdaRank算法进行股票排序
- **完全复用**: 复用现有特征工程、数据预处理、评估指标（final_score）
- **高效训练**: LGBM训练速度远快于Transformer（分钟级 vs 小时级）
- **可解释性**: 提供特征重要性分析

## 项目结构

```
LGBM/
├── config.py                    # LGBM专用配置（继承现有配置）
├── data_loader.py              # 数据加载与预处理（修正标签计算）
├── train_lgbm.py               # LGBM训练主脚本
├── evaluate_lgbm.py            # 模型评估脚本
├── predict_lgbm.py             # 预测脚本
├── requirements.txt            # 依赖包
└── README.md                   # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- lightgbm>=4.0.0
- scikit-learn>=1.3.0
- pandas>=2.0.0
- numpy>=1.24.0
- joblib>=1.3.0

### 2. 数据准备

确保数据文件位于 `../data/` 目录下：
- `train.csv`: 训练数据（2015-01-05 至 2025-03-06）
- `test.csv`: 测试数据（2025-03-09 至 2026-03-13）

### 3. 训练模型

```bash
python train_lgbm.py
```

训练过程：
1. 加载数据并应用修正后的标签计算
2. 创建扁平化数据集（每个(股票,日期)为独立样本）
3. 按最后一个月划分验证集
4. 训练LGBM LambdaRank模型
5. 评估模型并保存final_score

输出文件保存在 `LGBM/model/` 目录：
- `lgbm_model.pkl`: 训练好的模型
- `config.json`: 训练配置
- `feature_importance.csv`: 特征重要性排名
- `final_score.txt`: 验证集final_score
- `evaluation_results.json`: 详细评估结果

### 4. 评估模型

```bash
python evaluate_lgbm.py
```

重新计算验证集性能，生成详细分析报告。

### 5. 预测Top5股票

```bash
python predict_lgbm.py
```

使用最新数据预测未来一周收益率最高的5只股票。

## 配置说明

主要配置参数（`config.py`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_type` | `'ranking'` | 模型类型：'ranking'（LambdaRank） |
| `n_estimators` | 1000 | 树的数量 |
| `learning_rate` | 0.05 | 学习率 |
| `num_leaves` | 31 | 叶子数量 |
| `early_stopping_rounds` | 50 | 早停轮数 |
| `eval_metric` | `'ndcg'` | 评估指标 |
| `label_mode` | `'t1_to_t5'` | 标签计算模式 |

## 标签计算修正

**原Transformer标签**（存在问题）：
```
label = (open_{t+5} - open_t) / open_t
```
包含T→T+1的不可交易收益（隔夜跳空等噪声）。

**修正后LGBM标签**（与竞赛一致）：
```
label = (open_{t+5} - open_{t+1}) / open_{t+1}
```
仅包含T+1开盘到T+5开盘的可交易收益。

## 评估指标

完全复用Transformer的`final_score`计算：
- **Predicted Top5 Return Sum**: 模型预测的Top5股票实际收益和
- **Theoretical Max Return Sum**: 理论上最优的Top5股票收益和
- **Random Return Sum**: 随机选择的Top5股票期望收益和
- **Final Score**: `(Predicted - Random) / (Max - Random)`

## 特征工程

复用现有的197个特征（158个Alpha特征 + 39个技术指标）：
- 158个Alpha特征：动量、波动率、相关性等
- 39个技术指标：SMA、EMA、RSI、MACD、KDJ、布林带等

## 性能对比

| 指标 | Transformer | LGBM |
|------|------------|------|
| 训练时间 | 数小时 | 数分钟 |
| 内存占用 | 高 | 低 |
| 可解释性 | 低 | 高（特征重要性） |
| Final Score | 待对比 | 待评估 |

## 常见问题

### 1. 导入错误 "cannot import name 'lgbm_config'"
确保在LGBM目录下运行脚本，或正确设置Python路径。

### 2. 内存不足
LGBM数据集较大（约60万样本×197特征），建议确保有足够内存（≥8GB）。

### 3. 训练时间过长
调整`n_estimators`和`early_stopping_rounds`，或使用更少的特征。

### 4. Final Score为负
可能原因：
- 标签计算仍有问题
- 特征与收益率相关性弱
- 模型欠拟合/过拟合

## 后续优化建议

1. **特征选择**: 基于特征重要性筛选Top50特征
2. **超参数调优**: 使用网格搜索或贝叶斯优化
3. **模型集成**: 与Transformer模型集成（加权平均）
4. **时序交叉验证**: 使用滚动窗口交叉验证
5. **标签平滑**: 处理极端收益率样本

## 参考

- [LightGBM文档](https://lightgbm.readthedocs.io/)
- [Learning to Rank简介](https://en.wikipedia.org/wiki/Learning_to_rank)
- [竞赛Baseline代码](../code/src/)