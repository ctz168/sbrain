#!/bin/bash
# 快速启动Telegram Bot

TOKEN="${1:-7983263905:AAFsMuGRdZzWv7KfUaAkJocu0l7LsHrScuc}"
DEVICE="${2:-cpu}"

cd "$(dirname "$0")/.."

echo "启动类人脑双系统AI Bot..."
echo "Token: ${TOKEN:0:10}..."
echo "设备: $DEVICE"

python3 telegram_bot/run_bot.py --token "$TOKEN" --device "$DEVICE"
