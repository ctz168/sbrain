#!/usr/bin/env python3
"""
类人脑双系统AI - Telegram Bot 启动脚本 (完整版)

真正整合所有6大模块：
- 模块1：双轨权重原生改造
- 模块2：多尺度时序嵌套推理引擎
- 模块3：全链路STDP学习系统
- 模块4：元认知双闭环校验系统
- 模块5：海马体-新皮层协同记忆系统
- 模块6：多任务场景自适应预适配模块
"""

import os
import sys
import argparse

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(description='类人脑双系统AI Telegram Bot (完整版)')
    parser.add_argument('--token', type=str, required=True, help='Telegram Bot Token')
    parser.add_argument('--model-path', type=str, default='./models/DeepSeek-R1-Distill-Qwen-1.5B', help='模型路径')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='运行设备')
    parser.add_argument('--quantization', type=str, default='FP32', choices=['FP32', 'INT4', 'INT8'], help='量化类型')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("类人脑双系统全闭环AI架构 - Telegram Bot (完整版)")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"量化: {args.quantization}")
    print("=" * 60)
    
    # 检查模型路径
    model_path = args.model_path
    if not os.path.exists(model_path):
        alt_path = os.path.join(project_root, "models/DeepSeek-R1-Distill-Qwen-1.5B")
        if os.path.exists(alt_path):
            model_path = alt_path
            print(f"使用模型: {model_path}")
        else:
            print(f"\n❌ 模型未找到: {model_path}")
            print(f"请运行: huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./models/DeepSeek-R1-Distill-Qwen-1.5B")
            sys.exit(1)
    
    # 加载完整的类人脑AI接口
    print("\n正在初始化类人脑双系统AI...")
    try:
        from core.brain_interface import BrainAIInterface
        
        ai_interface = BrainAIInterface(
            model_path=model_path,
            device=args.device
        )
        
    except Exception as e:
        print(f"❌ 初始化失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 启动Bot
    print("\n正在启动Bot服务...")
    try:
        from telegram_bot.bot import BrainAIBot
        
        bot = BrainAIBot(token=args.token, ai_interface=ai_interface)
        print("✓ Bot 就绪")
        print("\n启动服务... (Ctrl+C 停止)")
        print("=" * 60)
        bot.run()
        
    except Exception as e:
        print(f"❌ Bot启动失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
