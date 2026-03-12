#!/bin/bash
# 类人脑双系统AI架构 - 部署脚本

set -e

echo "============================================================"
echo "类人脑双系统全闭环AI架构 - 部署脚本"
echo "============================================================"

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python版本: $PYTHON_VERSION"

# 安装依赖
echo ""
echo "[1/4] 安装依赖..."
pip install --break-system-packages -r requirements.txt

# 下载模型
echo ""
echo "[2/4] 下载模型..."
if [ ! -d "models/Qwen2.5-0.5B" ]; then
    echo "正在下载Qwen2.5-0.5B模型..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-0.5B', local_dir='./models/Qwen2.5-0.5B')"
else
    echo "模型已存在，跳过下载"
fi

# 验证安装
echo ""
echo "[3/4] 验证安装..."
python3 -c "
import torch
import transformers
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ Transformers {transformers.__version__}')
"

# 创建必要的目录
echo ""
echo "[4/4] 创建目录..."
mkdir -p weights logs

echo ""
echo "============================================================"
echo "部署完成！"
echo "============================================================"
echo ""
echo "使用方法:"
echo "  交互式对话: python main.py --mode chat"
echo "  启动Bot:    python telegram_bot/run_bot.py --token YOUR_TOKEN"
echo "  运行测评:   python main.py --mode eval"
echo ""
