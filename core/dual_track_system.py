#!/usr/bin/env python3
"""
双轨处理系统：逻辑轨 + 创意轨

核心功能：
1. 自动识别问题类型
2. 逻辑问题：逻辑链稠密化
3. 创意问题：发散思维激发
4. 混合问题：双轨融合
"""

import os
import sys
import time
import asyncio
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.problem_classifier import (
    ProblemTypeClassifier, 
    ProblemType, 
    ProblemAnalysis
)


@dataclass
class ProcessingResult:
    """处理结果"""
    content: str
    problem_type: ProblemType
    logic_score: float
    creative_score: float
    processing_track: str
    density: int  # 逻辑链稠密度（仅逻辑轨有意义）


class LogicTrack:
    """
    逻辑轨道：处理逻辑锚点强的问题
    
    特点：
    - 逻辑链稠密化
    - 符号计算辅助
    - 步骤验证
    """
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    async def process(self, question: str, analysis: ProblemAnalysis) -> Tuple[str, int]:
        """
        逻辑轨处理
        
        Returns:
            result: 处理结果
            density: 逻辑链稠密度
        """
        # 1. 构建稠密逻辑链提示
        prompt = self._build_dense_logic_prompt(question)
        
        # 2. 生成回答
        result = await self._generate(prompt, temperature=0.2)
        
        # 3. 计算稠密度
        density = self._calculate_density(result)
        
        return result, density
    
    def _build_dense_logic_prompt(self, question: str) -> str:
        """构建稠密逻辑链提示"""
        return f"""问题：{question}

请严格按照以下逻辑步骤进行分析：

═══════════════════════════════════════════════
【第一步：问题理解】
═══════════════════════════════════════════════
1.1 问题的核心是什么？
1.2 需要什么信息才能解决？
1.3 期望的输出格式是什么？

═══════════════════════════════════════════════
【第二步：信息提取】
═══════════════════════════════════════════════
2.1 有哪些关键数据/条件？
2.2 数据之间的关系是什么？
2.3 有哪些隐含的假设？

═══════════════════════════════════════════════
【第三步：推理链构建】
═══════════════════════════════════════════════
3.1 需要哪些推理步骤？
3.2 每步的依据是什么？
3.3 步骤之间的依赖关系是什么？

═══════════════════════════════════════════════
【第四步：执行推理】
═══════════════════════════════════════════════
4.1 逐步执行推理
4.2 记录中间结果
4.3 验证每步的正确性

═══════════════════════════════════════════════
【第五步：结果验证】
═══════════════════════════════════════════════
5.1 检查结果的合理性
5.2 检查单位/数量级
5.3 确认答案的完整性

═══════════════════════════════════════════════
【最终答案】
═══════════════════════════════════════════════
请给出明确的答案：

"""
    
    async def _generate(self, prompt: str, temperature: float = 0.3) -> str:
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.2
            )
        
        result = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return result
    
    def _calculate_density(self, text: str) -> int:
        """计算逻辑链稠密度"""
        # 统计推理步骤的数量
        steps = len(re.findall(r'【第\w+步', text))
        substeps = text.count('•') + text.count('-') + text.count('.')
        return steps * 5 + substeps


class CreativeTrack:
    """
    创意轨道：处理创意需求强的问题
    
    特点：
    - 发散思维
    - 联想扩展
    - 多角度探索
    """
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    async def process(self, question: str, analysis: ProblemAnalysis) -> str:
        """创意轨处理"""
        # 1. 构建创意激发提示
        prompt = self._build_creative_prompt(question)
        
        # 2. 高温度生成
        result = await self._generate(prompt, temperature=0.8)
        
        return result
    
    def _build_creative_prompt(self, question: str) -> str:
        """构建创意激发提示"""
        return f"""主题：{question}

请从以下多个角度自由发挥，展现创意：

╔═══════════════════════════════════════════════════════════╗
║                    【创意思维空间】                        ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  ◆ 角度一：核心主题                                       ║
║    - 这个主题的本质是什么？                               ║
║    - 有哪些可以深入探索的方向？                           ║
║                                                           ║
║  ◆ 角度二：联想扩展                                       ║
║    - 有哪些相关的概念或意象？                             ║
║    - 可以如何延伸和连接？                                 ║
║                                                           ║
║  ◆ 角度三：独特视角                                       ║
║    - 有什么新颖的切入点？                                 ║
║    - 如何让内容更有特色？                                 ║
║                                                           ║
║  ◆ 角度四：情感共鸣                                       ║
║    - 如何触动读者的心？                                   ║
║    - 有什么动人的元素？                                   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

请自由创作，展现你的想象力和创造力：

"""
    
    async def _generate(self, prompt: str, temperature: float = 0.8) -> str:
        """生成创意内容"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1
            )
        
        result = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return result


class HybridTrack:
    """
    混合轨道：处理逻辑和创意并存的问题
    
    特点：
    - 逻辑部分用逻辑轨
    - 创意部分用创意轨
    - 动态融合
    """
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logic_track = LogicTrack(model, tokenizer, device)
        self.creative_track = CreativeTrack(model, tokenizer, device)
    
    async def process(
        self, 
        question: str, 
        analysis: ProblemAnalysis
    ) -> Tuple[str, int]:
        """
        混合轨处理
        
        根据逻辑锚点和创意需求的比例动态调整
        """
        # 计算融合比例
        logic_ratio = analysis.logic_anchor_score / (
            analysis.logic_anchor_score + analysis.creative_score + 0.01
        )
        creative_ratio = 1 - logic_ratio
        
        # 构建混合提示
        prompt = self._build_hybrid_prompt(question, logic_ratio, creative_ratio)
        
        # 中等温度生成
        temperature = 0.3 + 0.4 * creative_ratio
        result = await self._generate(prompt, temperature)
        
        # 计算稠密度
        density = self._calculate_density(result)
        
        return result, density
    
    def _build_hybrid_prompt(
        self, 
        question: str, 
        logic_ratio: float,
        creative_ratio: float
    ) -> str:
        """构建混合提示"""
        logic_pct = int(logic_ratio * 100)
        creative_pct = int(creative_ratio * 100)
        
        return f"""问题：{question}

请结合逻辑分析和创意思考来回答：

┌─────────────────────────────────────────────────────────────┐
│                    分析框架                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【逻辑分析部分】({logic_pct}%权重)                         │
│  ─────────────────                                          │
│  • 事实依据是什么？                                         │
│  • 有哪些逻辑推理？                                         │
│  • 如何验证正确性？                                         │
│                                                             │
│  【创意思考部分】({creative_pct}%权重)                      │
│  ─────────────────                                          │
│  • 有什么独特的视角？                                       │
│  • 如何让内容更有价值？                                     │
│  • 有什么创新的思路？                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

请综合以上两个方面，给出完整的回答：

"""
    
    async def _generate(self, prompt: str, temperature: float = 0.5) -> str:
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.15
            )
        
        result = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return result
    
    def _calculate_density(self, text: str) -> int:
        """计算逻辑链稠密度"""
        steps = len(re.findall(r'【\w+】', text))
        return steps * 3


class DualTrackBrainAI:
    """
    双轨类脑AI系统
    
    核心创新：
    1. 自动识别问题类型
    2. 逻辑问题用逻辑轨（稠密化）
    3. 创意问题用创意轨（发散）
    4. 混合问题双轨融合
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        
        print("\n" + "=" * 60)
        print("双轨类脑AI系统")
        print("=" * 60)
        
        # 加载模型
        print("\n[1/4] 加载模型...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32,
            device_map={"": device}, trust_remote_code=True
        )
        self.model.eval()
        print(f"  ✓ 模型加载成功")
        
        # 问题分类器
        print("\n[2/4] 初始化问题分类器...")
        self.classifier = ProblemTypeClassifier()
        print("  ✓ 问题分类器初始化完成")
        
        # 三条轨道
        print("\n[3/4] 初始化处理轨道...")
        self.logic_track = LogicTrack(self.model, self.tokenizer, device)
        self.creative_track = CreativeTrack(self.model, self.tokenizer, device)
        self.hybrid_track = HybridTrack(self.model, self.tokenizer, device)
        print("  ✓ 三条轨道初始化完成")
        
        # 统计
        print("\n[4/4] 初始化统计模块...")
        self.stats = {
            'total_queries': 0,
            'logic_queries': 0,
            'creative_queries': 0,
            'hybrid_queries': 0
        }
        print("  ✓ 统计模块初始化完成")
        
        print("\n" + "=" * 60)
        print("✓ 双轨系统初始化完成")
        print("=" * 60)
    
    async def generate_stream(self, input_text: str, max_tokens: int = 300):
        """
        流式生成
        
        根据问题类型自动选择处理轨道
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        # 1. 分析问题类型
        analysis = self.classifier.classify(input_text)
        
        print(f"\n[问题分析]")
        print(f"  类型: {analysis.problem_type.value}")
        print(f"  逻辑锚点: {analysis.logic_anchor_score:.2f}")
        print(f"  创意需求: {analysis.creative_score:.2f}")
        
        # 2. 根据类型选择轨道
        if analysis.problem_type in [ProblemType.PURE_LOGIC, ProblemType.LOGIC_DOMINANT]:
            # 逻辑轨
            result, density = await self.logic_track.process(input_text, analysis)
            track = "logic"
            self.stats['logic_queries'] += 1
        elif analysis.problem_type == ProblemType.PURE_CREATIVE:
            # 创意轨
            result = await self.creative_track.process(input_text, analysis)
            density = 0
            track = "creative"
            self.stats['creative_queries'] += 1
        else:
            # 混合轨
            result, density = await self.hybrid_track.process(input_text, analysis)
            track = "hybrid"
            self.stats['hybrid_queries'] += 1
        
        # 3. 流式输出
        for char in result:
            yield char
            await asyncio.sleep(0.01)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'total_queries': self.stats['total_queries'],
            'logic_queries': self.stats['logic_queries'],
            'creative_queries': self.stats['creative_queries'],
            'hybrid_queries': self.stats['hybrid_queries']
        }
