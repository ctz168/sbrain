#!/usr/bin/env python3
"""
LSDC 引擎 - 主入口

逻辑自相似稠密补齐 (Logic Self-similar Dense Completion) 引擎

数学原理：
1. 离散状态转移: S_n → S_{n+1}
2. 窄宽带补齐: f(S_n, Δt) → S_{n+1}
3. 自相似结构: [前提, 推演, 结论]
"""

import os
import sys
import argparse
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsdc_engine.logic_processor import create_logic_processor, LogicProcessor
from lsdc_engine.model_handler import create_model_handler


def print_banner():
    """打印横幅"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██╗     ███████╗ ██████╗ ██████╗ ███████╗                  ║
║   ██║     ██╔════╝██╔═══██╗██╔══██╗██╔════╝                  ║
║   ██║     ███████╗██║   ██║██║  ██║█████╗                    ║
║   ██║     ╚════██║██║   ██║██║  ██║██╔══╝                    ║
║   ███████╗███████║╚██████╔╝██████╔╝███████╗                  ║
║   ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝                  ║
║                                                               ║
║   Logic Self-similar Dense Completion Engine                 ║
║   逻辑自相似稠密补齐引擎                                      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")


def print_math_model():
    """打印数学模型"""
    print("""
═══════════════════════════════════════════════════════════════════
                        核心数学模型
═══════════════════════════════════════════════════════════════════

1. 离散状态转移
   ─────────────────────────────────────────────────────────────
   S_n ──f()──> S_{n+1}
   
   每个逻辑节点 S_n 代表一个离散的思维状态

2. 窄宽带补齐
   ─────────────────────────────────────────────────────────────
   f(S_n, Δt) → S_{n+1}
   
   • 每次只喂入上一个微步的 Conclusion
   • 和当前的目标 Goal
   • 丢弃所有历史过程，防止模型过载

3. 自相似结构
   ─────────────────────────────────────────────────────────────
   [前提, 推演, 结论] 三位一体
   
   微观尺度: 单个推理步
   宏观尺度: 完整推理链
   结构同构: 任意尺度的结构保持一致

═══════════════════════════════════════════════════════════════════
""")


async def interactive_mode(processor: LogicProcessor):
    """交互模式"""
    print("\n" + "="*60)
    print("交互模式 (输入 'quit' 退出)")
    print("="*60)
    
    context = None
    
    while True:
        print("\n" + "-"*60)
        goal = input("请输入问题: ").strip()
        
        if goal.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        
        if not goal:
            continue
        
        print(f"\n处理: {goal}")
        print("-"*60)
        
        # 处理
        for node in processor.process(goal, context):
            print(f"\n[节点 {node.node_id}] 密度: {node.density:.1f}")
            if node.premise:
                print(f"  前提: {node.premise[:50]}...")
            if node.derivation:
                print(f"  推演: {node.derivation[:80]}...")
            if node.conclusion:
                print(f"  结论: {node.conclusion[:50]}...")
        
        # 获取逻辑链
        chain = processor.get_chain()
        if chain:
            print("\n" + "="*60)
            print("完整逻辑链:")
            print("="*60)
            print(chain.to_text())
            
            # 更新上下文
            context = chain.get_last_conclusion()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LSDC 引擎")
    parser.add_argument("--mode", choices=["interactive", "server", "test"], default="interactive")
    parser.add_argument("--model", type=str, default="../models/Qwen3.5-0.8B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--goal", type=str, help="直接处理目标问题")
    
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    print_math_model()
    
    # 创建逻辑处理器
    print("初始化 LSDC 引擎...")
    processor = create_logic_processor(args.model, args.device)
    
    if args.mode == "interactive":
        # 交互模式
        asyncio.run(interactive_mode(processor))
    
    elif args.mode == "server":
        # 服务器模式
        import uvicorn
        os.environ["MODEL_PATH"] = args.model
        os.environ["DEVICE"] = args.device
        uvicorn.run("app:app", host="0.0.0.0", port=args.port)
    
    elif args.mode == "test":
        # 测试模式
        test_goal = args.goal or "3月份20天房租1600元，月租是多少？"
        print(f"\n测试问题: {test_goal}")
        print("-"*60)
        
        for node in processor.process(test_goal):
            print(f"\n[节点 {node.node_id}]")
            print(f"  前提: {node.premise[:50]}...")
            print(f"  推演: {node.derivation[:80]}...")
            print(f"  结论: {node.conclusion[:50]}...")
        
        chain = processor.get_chain()
        if chain:
            print("\n完整逻辑链:")
            print(chain.to_text())


if __name__ == "__main__":
    main()
