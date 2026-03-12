#!/bin/bash
# LSDC 引擎环境安装脚本

echo "============================================================"
echo "LSDC 引擎 - 环境安装"
echo "============================================================"

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"

# 安装依赖
echo ""
echo "[1/3] 安装Python依赖..."
pip install -r requirements.txt

# 检查模型
echo ""
echo "[2/3] 检查模型..."
if [ ! -d "../models/Qwen3.5-0.8B" ]; then
    echo "警告: 未找到Qwen3.5-0.8B模型"
    echo "请运行以下命令下载模型:"
    echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', local_dir='../models/Qwen3.5-0.8B')\""
else
    echo "✓ 模型已存在"
fi

# 验证安装
echo ""
echo "[3/3] 验证安装..."
python3 -c "
import torch
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print('✓ 环境验证通过')
"

echo ""
echo "============================================================"
echo "✓ 安装完成"
echo "============================================================"
