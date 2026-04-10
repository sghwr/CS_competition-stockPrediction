#!/bin/bash

# LightGBM回归模型训练与预测脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "================================================"
echo "LightGBM股票收益率预测"
echo "================================================"

# 检查依赖
echo "检查Python依赖..."
python3 -c "import lightgbm" 2>/dev/null || {
    echo "错误: LightGBM未安装"
    echo "请安装: pip install lightgbm"
    exit 1
}

python3 -c "import pandas" 2>/dev/null || {
    echo "错误: pandas未安装"
    exit 1
}

# 创建必要目录
mkdir -p "$SCRIPT_DIR/model"
mkdir -p "$SCRIPT_DIR/output"

# 检查数据文件
DATA_PATH="$PROJECT_ROOT/data/train.csv"
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    exit 1
fi
export DATA_PATH

echo "数据文件: $DATA_PATH"
echo "工作目录: $SCRIPT_DIR"

# 菜单选择
echo ""
echo "请选择操作:"
echo "1. 训练模型"
echo "2. 运行预测"
echo "3. 训练并预测"
echo "4. 退出"
echo -n "请输入选择 (1-4): "
read choice

case $choice in
    1)
         echo "开始训练模型..."
         cd "$SCRIPT_DIR"
         python3 train_lgbm.py "$DATA_PATH"
        ;;
    2)
         echo "开始运行预测..."
         cd "$SCRIPT_DIR"
         python3 predict_lgbm.py "$DATA_PATH"
        ;;
    3)
         echo "开始训练并预测..."
         cd "$SCRIPT_DIR"
         python3 train_lgbm.py "$DATA_PATH"
         echo ""
         echo "训练完成，开始预测..."
         python3 predict_lgbm.py "$DATA_PATH"
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "操作完成!"
echo "================================================"