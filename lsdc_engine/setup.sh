#!/bin/bash
# setup.sh - LSDC引擎环境安装脚本

echo "============================================================"
echo "LSDC引擎环境安装"
echo "============================================================"

# 检查Python版本
echo "[1/5] 检查Python版本..."
python3 --version

# 安装依赖
echo "[2/5] 安装Python依赖..."
pip install torch transformers fastapi uvicorn sse-starlette redis --quiet

# 创建必要目录
echo "[3/5] 创建目录结构..."
mkdir -p core
mkdir -p models
mkdir -p logs

# 检查模型
echo "[4/5] 检查模型..."
if [ ! -d "models/Qwen3.5-0.8B" ]; then
    echo "请先下载Qwen3.5-0.8B模型"
    echo "运行: python3 -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B').save_pretrained('models/Qwen3.5-0.8B')\""
else
    echo "模型已存在"
fi

# 设置权限
echo "[5/5] 设置权限..."
chmod +x push_to_git.sh

echo "============================================================"
echo "安装完成！"
echo "============================================================"
