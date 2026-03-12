#!/usr/bin/env python3
"""
LSDC 引擎 - 5轮互动测试

模拟用户与Bot的5轮对话
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsdc_engine.logic_processor import create_logic_processor, LogicProcessor


async def test_5_rounds():
    """5轮互动测试"""
    
    print("\n" + "=" * 70)
    print("LSDC 引擎 - 5轮互动测试")
    print("=" * 70)
    
    # 创建逻辑处理器
    print("\n初始化LSDC引擎...")
    processor = create_logic_processor(
        model_path="../models/Qwen3.5-0.8B",
        device="cpu"
    )
    
    # 5轮对话
    conversations = [
        {
            "round": 1,
            "user": "3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。月租是多少？",
            "expected": "计算月租"
        },
        {
            "round": 2,
            "user": "卫生费什么时候退",
            "expected": "回答退费条件"
        },
        {
            "round": 3,
            "user": "押金怎么退",
            "expected": "回答押金退还"
        },
        {
            "round": 4,
            "user": "月租可以退吗",
            "expected": "回答月租是否可退"
        },
        {
            "round": 5,
            "user": "日租是多少",
            "expected": "计算日租"
        }
    ]
    
    # 上下文
    context = None
    
    # 执行测试
    for conv in conversations:
        print("\n" + "=" * 70)
        print(f"第 {conv['round']} 轮对话")
        print("=" * 70)
        print(f"\n👤 用户: {conv['user']}")
        print(f"   预期: {conv['expected']}")
        print("-" * 70)
        
        # 处理
        node_count = 0
        last_conclusion = None
        
        for node in processor.process(conv['user'], context):
            node_count += 1
            
            print(f"\n[节点 {node.node_id}] 密度: {node.density:.1f}")
            if node.premise:
                print(f"  前提: {node.premise[:60]}...")
            if node.derivation:
                print(f"  推演: {node.derivation[:80]}...")
            if node.conclusion:
                print(f"  结论: {node.conclusion[:60]}...")
                last_conclusion = node.conclusion
        
        # 更新上下文
        context = last_conclusion
        
        print(f"\n📊 本轮统计:")
        print(f"   节点数: {node_count}")
        print(f"   上下文: {context[:40] if context else 'None'}...")
    
    # 最终统计
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    
    chain = processor.get_chain()
    if chain:
        print(f"\n完整逻辑链 ({len(chain.nodes)} 个节点):")
        print("-" * 70)
        print(chain.to_text()[:1000])
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(test_5_rounds())
