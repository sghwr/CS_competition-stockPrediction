
> THU-BigDataCompetition-2026-ver4 — 多模型集成排序学习选股方案
>
> 在原始 StockTransformer 基线基础上，引入 LightGBM 排序模型，最终通过 加权集成（LGBM 0.4 + Transformer 0.6 + Z-score 归一化） 融合两者 logits，在测试集滑动窗口回测中实现 *年化 Sharpe 3.07、胜率 57.7%、最大回撤 -18.6%*。
第 1 节：项目目标与整体流程
- 描述问题：沪深 300 成分股排序选股
- ver4 的整体流程：特征工程 → 三套模型分别训练/评估 → 集成 → 滑动窗口回测
第 2 节：整体架构
数据层      stock_data.csv → 特征工程 (158+39维)
            ↓
模型层      Transformer (code/src/)    LGBM (LGBM/)
            ↓                          ↓
集成层      ensemble/core.py → Z-score归一化 → 加权融合
            ↓
评估层      test/eval_sliding_window.py (Transformer)
            test/eval_lgbm.py (LGBM)
            ensemble/evaluate.py (Ensemble)
            widget/testflight.py (非重叠窗口修正)
第 3 节：代码结构说明
模块
基线 Transformer
LGBM 排序模型
集成模型
滑动窗口评估
修正回测
数据
第 4 节：特征工程
- 统一使用 158+39 特征集（197 维）
- 描述 158 个 Alpha 因子（含 ROC、MA、STD、BETA、RSQR、MAX/MIN、CORR 等 5/10/20/30/60 五组窗口）
- 描述 39 个技术指标（SMA、EMA、MACD、RSI、KDJ、BOLL、ATR、OBV 等）
- 基于 TA-Lib 实现，按股票独立分组、向后看窗口、无前瞻偏差
第 5 节：各模型说明
5.1 Transformer（基线）
- StockTransformer 架构：PositionalEncoding → TransformerEncoder → FeatureAttention → CrossStockAttention → score_head
- MD-SRP 模块（可选行业动量先验）
- 训练：ListNet + Pairwise 组合损失，监控 final_score
- 产出: model/60_158+39/best_model.pth
5.2 LGBM
- LambdaRank 排序目标（lambdarank）
- Flat 特征输入（197 维，无需标准化）
- 训练：滑窗构建样本、日期连续性过滤
- 产出: LGBM/model/lgbm_model.pkl
5.3 Ensemble 集成
- 策略：逐日分别对两模型推理 → Z-score 归一化 → 0.6 × TF + 0.4 × LGBM
- 评估方式：滑动窗口回测，每日选 Top-5，等权持有 5 个交易日，计算真实持有期收益率
第 6 节：训练与评估流程
1. 训练 Transformer：cd code/src && python train.py
2. 训练 LGBM：cd LGBM && python train_lgbm.py
3. 单模型评估：python test/eval_sliding_window.py / python test/eval_lgbm.py
4. 集成评估：python -m ensemble.evaluate
5. 生成预测：python -m ensemble.predict → ensemble/output/result.csv
第 7 节：评估指标
指标
年化 Sharpe
胜率
盈亏比
最大回撤
累计收益率
第 8 节：实验结果
模型
Transformer
LGBM
Ensemble (TF 0.6 + LGBM 0.4)
第 9 节：运行环境
- Python 3.11+
- PyTorch / LightGBM / TA-Lib / pandas / numpy / scikit-learn / joblib
---
