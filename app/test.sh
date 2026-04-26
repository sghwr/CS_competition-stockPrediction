#!/bin/bash
set -e

echo "=== 开始测试/预测 ==="
cd /app

echo "运行预测..."
python code/src/predict_ensemble.py

echo "预测完成"