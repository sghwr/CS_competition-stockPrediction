# LightGBM股票收益率预测实验

## 项目概述

本实验使用LightGBM回归模型直接预测股票未来5日收益率，替代原baseline的排序学习（Learning to Rank）框架。目标是解决排序学习与收益率预测任务之间的错配问题。

### 核心改进
1. **任务目标**：从排序学习改为回归预测
2. **损失函数**：从排序损失改为MSE回归损失  
3. **评估标准**：直接评估收益率预测准确性
4. **特征工程**：复用原baseline的158+39个技术指标特征

## 问题分析

原baseline使用排序学习框架，存在以下问题：
- 模型分数与收益率相关性弱（Spearman=0.106）
- Top5命中率为0%
- 模型错过真正的高收益股票（0.217平均收益 vs 0.025模型选择）

本实验通过回归模型直接预测收益率数值，为权重分配提供可靠依据。

## 文件结构

```
LGBM-testflight/
├── train_lgbm.py          # 训练脚本
├── predict_lgbm.py        # 预测脚本
├── run.sh                 # 运行脚本
├── README.md              # 说明文档
├── model/                 # 模型保存目录
│   ├── lgbm_model.pkl     # 训练好的模型
│   ├── feature_columns.pkl # 特征列名
│   └── ...               # 其他模型文件
└── output/                # 输出目录
    ├── result.csv         # 预测结果（Top5股票）
    └── full_predictions.csv # 完整预测结果
```

## 快速开始

### 1. 安装依赖

```bash
# 确保已安装基础依赖（同原项目）
# 安装LightGBM
pip install lightgbm
```

### 2. 运行实验

```bash
cd #myWorkfold/LGBM-testflight

# 赋予执行权限
chmod +x run.sh

# 运行交互式菜单
./run.sh
```

或直接运行：

```bash
# 仅训练
python train_lgbm.py

# 仅预测
python predict_lgbm.py
```

## 模型详情

### 特征工程
复用原baseline的`engineer_features_158plus39`函数，包含：
- 158个Alpha特征
- 39个技术指标特征（SMA、EMA、RSI、MACD、KDJ等）

### 数据预处理
1. 加载`data/train.csv`
2. 按股票分组进行特征工程
3. 构建标签：未来5日收益率 `(open_t5 - open_t1) / open_t1`
4. 过滤无效数据（开盘价过小、NaN值）

### 模型架构
- **算法**：LightGBM Gradient Boosting
- **任务**：回归（预测收益率）
- **参数**：
  - `objective: 'regression'`
  - `metric: 'mse'`
  - `num_leaves: 31`
  - `learning_rate: 0.05`
  - `num_iterations: 1000`
  - `early_stopping_round: 50`

### 训练策略
1. **时间序列划分**：按日期划分训练/验证/测试集（避免数据泄露）
2. **验证集**：最后20%数据作为测试集
3. **早停机制**：防止过拟合
4. **特征重要性分析**：输出Top10重要特征

## 输出结果

### 训练输出
1. **模型文件**：`model/lgbm_model.pkl`
2. **特征列**：`model/feature_columns.pkl`
3. **评估结果**：`model/evaluation_results.csv`
4. **特征重要性**：`model/feature_importance.csv`

### 预测输出
1. **Top5股票**：`output/result.csv`（格式符合比赛要求）
2. **完整预测**：`output/full_predictions.csv`（所有股票预测结果）

### 评估指标
训练过程中输出：
1. **回归指标**：MSE、MAE、RMSE
2. **排名相关性**：预测排名与真实排名相关性
3. **Top5命中率**：预测Top5与真实Top5重合度
4. **超额收益**：模型Top5 vs 随机策略收益对比

## 与原Baseline对比

| 维度 | 原Baseline (排序学习) | 本实验 (回归预测) |
|------|---------------------|-------------------|
| **任务目标** | 相对排名正确 | 绝对收益率预测 |
| **损失函数** | WeightedRankingLoss | MSE回归损失 |
| **输出** | 任意尺度分数 | 收益率预测值 |
| **评估** | final_score排名相关性 | 收益率预测准确率 |
| **Top5依据** | 排序分数高低 | 预测收益率高低 |
| **权重分配** | 等权重0.2 | 等权重0.2（可扩展） |

## 预期优势

1. **直接优化比赛目标**：预测收益率直接对应比赛评估指标
2. **可解释性**：预测值有明确经济意义（预期收益率）
3. **扩展性**：预测值可直接用于权重优化
4. **计算效率**：LightGBM训练预测速度快

## 扩展方向

### 1. 权重优化
基于预测收益率设计权重分配函数：
```python
# 示例：softmax权重分配
weights = softmax(pred_returns[:5] / temperature)
```

### 2. 特征增强
增加更多预测因子：
- 市场资金流向
- 情绪指标
- 宏观因子

### 3. 模型集成
结合多个模型：
- LightGBM回归
- Transformer回归
- 传统时序模型

### 4. 风险管理
引入阈值机制：
```python
if max(pred_returns) < threshold:
    return []  # 空仓，持有现金
```

## 注意事项

1. **数据泄露**：确保特征工程不引入未来信息
2. **过拟合风险**：使用早停和正则化
3. **市场变化**：模型可能需定期重新训练
4. **计算资源**：特征工程计算量较大

## 故障排除

### 常见问题
1. **TA-Lib安装失败**
   ```bash
   # macOS
   brew install ta-lib
   pip install TA-Lib
   ```

2. **内存不足**
   - 减少特征数量
   - 使用数据采样
   - 增加`min_data_in_leaf`参数

3. **预测结果不稳定**
   - 增加训练数据量
   - 调整模型参数
   - 添加特征选择

### 日志输出
训练过程输出详细日志，包括：
- 数据加载进度
- 特征工程状态
- 训练迭代信息
- 评估指标结果

## 参考文献

1. Ke, G. et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NIPS 2017.
2. 原项目README：`THU-BDC2026/README.md`
3. 相关性分析报告：`#myWorkfold/stock_prediction_analysis_report.md`

## 联系方式

如有问题，请参考原项目文档或提交issue。

---
**实验目标**：验证回归模型在收益率预测任务上的有效性  
**数据范围**：沪深300成分股，2015-2026年  
**更新时间**：2026年4月10日