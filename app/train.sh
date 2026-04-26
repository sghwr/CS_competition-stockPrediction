#!/bin/bash
set -e

echo "=== 开始训练 ==="
cd /app

echo "训练Transformer模型..."
python code/src/train.py

echo "训练LGBM模型..."
python code/src/train_lgbm.py

echo "训练完成"