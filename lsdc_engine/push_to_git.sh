#!/bin/bash
# LSDC 引擎 - 自动化Git推送脚本

echo "============================================================"
echo "LSDC 引擎 - Git 自动化推送"
echo "============================================================"

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 检查是否有更改
if git diff-index --quiet HEAD --; then
    echo "没有需要提交的更改"
    exit 0
fi

# 显示更改
echo ""
echo "待提交的更改:"
echo "------------------------------------------------------------"
git status --short
echo "------------------------------------------------------------"

# 生成提交信息
COMMIT_MSG="feat(lsdc): 更新LSDC引擎

$(date '+%Y-%m-%d %H:%M:%S')

更新内容:
$(git diff --stat)"

# 添加所有更改
echo ""
echo "[1/3] 添加更改..."
git add -A

# 提交
echo ""
echo "[2/3] 提交更改..."
git commit -m "$COMMIT_MSG"

# 推送
echo ""
echo "[3/3] 推送到远程..."
git push origin main

echo ""
echo "============================================================"
echo "✓ Git 推送完成"
echo "============================================================"
