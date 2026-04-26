# 代码说明

## 环境配置

- Python: 3.12
- PyTorch: 2.6.0+
- LightGBM: 4.6.0+
- TA-Lib: 0.6.8
- pandas: 2.3.2+
- numpy: Latest
- scikit-learn: 1.7.2+
- tensorboardX: 2.6.4+
- joblib: 1.5.2+

## 数据

使用沪深300(HS300)成分股历史交易数据，数据来源为tushare/Akshare等金融数据API。
数据包括：
- 原始OHLCV数据（开盘价、收盘价、最高价、最低价、成交量、成交额）
- 股票行业分类数据（申万一级行业）
- 指数数据（用于市场状态划分）

训练数据：`/app/data/train.csv`
测试数据：`/app/data/test.csv`

## 预训练模型

本项目不使用预训练模型，采用从零训练的方式。

## 算法

### 整体思路介绍

本项目采用StockTransformer模型进行股票排序选股任务。核心思路是：
1. 对每只股票的历史价格数据进行特征工程，提取158+39=197维特征
2. 将股票排序问题建模为ListNet排序问题
3. 使用Transformer编码器捕捉股票间的时序依赖关系
4. 结合MD-SRP(Market-Driven Rank Prior)模块融入市场行业动量先验

### 方法的创新点

1. **MD-SRP模块**：基于市场状态（牛市/熊市/震荡/轮动）和行业动量先验，动态调整排序分数
2. **分层市场标准化**：根据每日市场收益率划分市场状态，对不同状态分别标准化收益率标签
3. **加权排序损失**：Top-5样本给予更高权重，强化模型对高收益股票的识别能力
4. **自适应先验权重网络**：根据市场状态动态学习行业先验的融合权重

### 网络结构

```
StockTransformer:
├── InputProjection (197 → 256)
├── PositionalEncoding
├── TransformerEncoder (3 layers, 4 heads)
├── FeatureAttention (序列维度加权聚合)
├── CrossStockAttention (股票间关系建模)
├── RankingLayers (256 → 128)
└── ScoreHead (128 → 1)
```

### 损失函数

采用组合损失函数：
- **Listwise Loss**: 加权交叉熵损失（KL散度形式）
- **Pairwise Loss**: 加权对比损失

最终损失：`L = L_listwise + λ * L_pairwise`，其中λ=1

Top-5样本权重为2.0，非Top-5样本权重为1.0

### 数据扩增

- 时间序列滑动窗口生成训练样本
- 标签winsorize处理（1%-99%分位数截断）
- 波动率标准化标签

### 模型集成

本项目为单模型方案，未使用模型集成。最终产出为best_model.pth

### 算法的其他细节

- 序列长度：60个交易日
- 验证集划分：最后两个月作为验证集
- 优化器：AdamW，lr=1e-5，weight_decay=1e-5
- 学习率调度：LinearLR，end_factor=0.2
- 梯度裁剪：max_norm=5.0
- 早停机制：基于final_score

## 训练流程

1. **数据加载**：从`/app/data/train.csv`加载原始OHLCV数据
2. **数据划分**：按时间划分训练集（最后两个月之前）和验证集（最后两个月）
3. **特征工程**：对每只股票独立计算158+39维特征（多进程加速）
4. **标签构建**：计算T+1到T+5开盘收益率作为标签
5. **标准化**：StandardScaler标准化特征，MarketNormalizer标准化标签
6. **数据集构建**：按日期组织样本，每个样本包含当日所有股票的特征序列
7. **模型训练**：50 epochs训练，监控final_score，保存最佳模型
8. **模型保存**：保存best_model.pth、scaler.pkl、market_normalizer.pkl

## 推理流程

1. 加载训练好的模型和标准化器
2. 对最新数据进行特征工程
3. 为每只股票构建60日历史序列
4. 模型前向传播得到排序分数
5. 选取Top-5股票，等权分配0.2权重
6. 输出结果到`/app/output/result.csv`

## 其他注意事项

- 验证数据划分时会保留序列上下文，确保验证集第一个样本有足够历史数据
- 训练时会过滤数据点不足的股票（少于60个交易日）
- 特征工程使用talib库加速，必须先安装TA-Lib C库
- Docker环境中使用NVIDIA GPU加速训练

## 文件结构

```
app/
├── code/
│   └── src/
│       ├── featurework.py    # 特征工程（158+39维技术指标）
│       ├── train.py          # 训练脚本
│       ├── test.py           # 推理脚本
│       ├── config.py         # 配置文件
│       ├── model.py          # StockTransformer模型
│       ├── market_normalizer.py  # 市场标准化器
│       └── market_prior.py   # MD-SRP模块
├── data/                     # 数据目录（docker-compose挂载）
│   ├── train.csv            # 训练数据
│   ├── test.csv             # 测试数据
│   ├── stock_industry.csv   # 行业数据
│   └── index_data.csv       # 指数数据
├── model/                    # 模型输出目录
│   ├── best_model.pth       # 最佳模型权重
│   ├── scaler.pkl           # 特征标准化器
│   └── market_normalizer.pkl # 市场标准化器
├── output/                   # 输出目录（docker-compose挂载）
│   └── result.csv           # 预测结果
├── temp/                     # 临时目录（docker-compose挂载）
├── init.sh                   # 初始化脚本
├── train.sh                  # 训练脚本
├── test.sh                   # 测试脚本
├── Dockerfile                # Docker镜像构建文件
└── docker-compose.yml       # Docker编排文件
```

## 快速开始

```bash
# 1. 构建Docker镜像
docker build -t bdc2026:latest -f app/Dockerfile app/

# 2. 使用docker-compose运行
docker-compose -f app/docker-compose.yml up

# 3. 在容器中执行训练
docker exec -it <container_id> bash /app/train.sh

# 4. 执行预测
docker exec -it <container_id> bash /app/test.sh

# 5. 导出镜像
docker save bdc2026:latest -o 队伍名称.tar
```