#!/bin/bash
set -e

echo "=== 初始化环境 ==="
cd /app

echo "安装依赖..."
uv sync --frozen

echo "创建必要的目录..."
mkdir -p /app/model
mkdir -p /app/output
mkdir -p /app/temp

echo "初始化完成"