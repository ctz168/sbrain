#!/usr/bin/env python3
"""
稠密逻辑链类脑AI接口

整合：
1. 分形推理引擎（自相似逻辑链稠密化）
2. 现有6大模块
3. 工具学习层

这是真正智能的实现方式！
"""

import os
import sys
import time
import asyncio
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.config import default_config
from core.fractal_reasoning import FractalReasoningEngine, SelfSimilarLogicChain, LogicNode


class DenseLogicBrainAI:
    """
    稠密逻辑链类脑AI
    
    核心创新：
    1. 自相似逻辑链稠密化
    2. 让模型自己展开推理步骤
    3. 无限深度的推理能力
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.config = default_config
        self.device = device
        
        print("\n" + "=" * 60)
        print("稠密逻辑链类脑AI架构")
        print("=" * 60)
        
        # ========== 1. 加载基础模型 ==========
        print("\n[1/5] 加载基础模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": device},
            trust_remote_code=True
        )
        self.base_model.eval()
        print(f"  ✓ 模型加载成功，词表大小：{len(self.tokenizer)}")
        
        # ========== 2. 分形推理引擎 ==========
        print("\n[2/5] 初始化分形推理引擎...")
        self.fractal_engine = FractalReasoningEngine(
            hidden_size=self.base_model.config.hidden_size,
            max_depth=3
        )
        print("  ✓ 分形推理引擎初始化完成")
        
        # ========== 3. 自相似逻辑链系统 ==========
        print("\n[3/5] 初始化自相似逻辑链系统...")
        self.logic_chain = SelfSimilarLogicChain(
            self.base_model,
            self.tokenizer,
            device
        )
        print("  ✓ 自相似逻辑链系统初始化完成")
        
        # ========== 4. 类脑模块 ==========
        print("\n[4/5] 初始化类脑模块...")
        try:
            from stdp.stdp_engine import STDPController
            from hippocampus.hippocampus_system import HippocampusSystem
            from metacognition.metacognition_system import MetacognitionSystem
            
            self.stdp = STDPController(self.config.stdp)
            self.hippocampus = HippocampusSystem(self.config.hippocampus)
            self.metacognition = MetacognitionSystem(
                self.config.metacognition,
                self.hippocampus,
                self.stdp
            )
            print("  ✓ 类脑模块初始化完成")
        except Exception as e:
            print(f"  ! 类脑模块初始化跳过: {e}")
            self.stdp = None
            self.hippocampus = None
            self.metacognition = None
        
        # ========== 5. 工具层 ==========
        print("\n[5/5] 初始化工具层...")
        self.tools = {
            'calculator': self._calculator_tool,
            'extractor': self._extractor_tool,
        }
        print("  ✓ 工具层初始化完成")
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'total_time': 0.0,
            'avg_density': 0.0,
            'total_nodes': 0
        }
        
        print("\n" + "=" * 60)
        print("✓ 所有模块初始化完成")
        print("=" * 60)
    
    async def generate_stream(self, input_text: str, max_tokens: int = 200):
        """
        稠密逻辑链推理
        
        核心流程：
        1. 模型理解问题，生成初始逻辑链
        2. 自相似展开，稠密化逻辑链
        3. 执行每一步推理
        4. 验证和输出
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        # ========== 第一步：模型理解并生成初始逻辑链 ==========
        initial_chain = await self._generate_initial_chain(input_text)
        
        # ========== 第二步：自相似展开稠密化 ==========
        dense_chain = self._densify_chain(initial_chain, input_text)
        
        # ========== 第三步：执行稠密逻辑链 ==========
        result = await self._execute_dense_chain(dense_chain, input_text)
        
        # ========== 第四步：记忆存储 ==========
        if self.hippocampus:
            self._store_to_memory(dense_chain, result)
        
        # 更新统计
        elapsed = time.time() - start_time
        self.stats['total_time'] += elapsed
        self.stats['avg_density'] = (
            (self.stats['avg_density'] * (self.stats['total_queries'] - 1) + 
             dense_chain.get('density', 1)) / self.stats['total_queries']
        )
        
        # 流式输出
        for char in result:
            yield char
            await asyncio.sleep(0.01)
    
    async def _generate_initial_chain(self, question: str) -> Dict:
        """
        让模型生成初始逻辑链
        
        这是关键：让模型自己思考需要哪些步骤
        """
        prompt = f"""问题：{question}

请分析解决这个问题需要哪些推理步骤？
请按以下格式列出：

步骤1：[步骤名称]
  - 子步骤1.1：[具体操作]
  - 子步骤1.2：[具体操作]
步骤2：[步骤名称]
  - 子步骤2.1：[具体操作]
...

推理步骤：
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.5,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 解析逻辑链
        chain = self._parse_logic_chain(response)
        
        return chain
    
    def _densify_chain(self, chain: Dict, context: str) -> Dict:
        """
        稠密化逻辑链
        
        核心思想：自相似展开
        每个步骤都可以展开为更细的子步骤
        """
        def expand_step(step: Dict, depth: int = 0) -> Dict:
            if depth >= 3:  # 最大深度
                return step
            
            # 自相似展开模板
            expand_templates = {
                '理解': ['识别关键信息', '确定问题类型', '明确目标'],
                '提取': ['定位数据', '识别单位', '确认关系'],
                '计算': ['建立公式', '代入数值', '执行运算', '检查结果'],
                '验证': ['检查单位', '检查数量级', '检查逻辑'],
                '输出': ['组织语言', '格式化', '最终确认']
            }
            
            step_name = step.get('name', '')
            
            # 找到匹配的展开模板
            for key, substeps in expand_templates.items():
                if key in step_name:
                    step['children'] = [
                        expand_step({'name': s, 'depth': depth + 1}, depth + 1)
                        for s in substeps
                    ]
                    break
            
            return step
        
        # 对每个步骤进行展开
        if 'steps' in chain:
            chain['steps'] = [expand_step(s) for s in chain['steps']]
        
        # 计算稠密度
        chain['density'] = self._calculate_density(chain)
        
        return chain
    
    async def _execute_dense_chain(self, chain: Dict, question: str) -> str:
        """
        执行稠密逻辑链
        """
        # 构建详细的推理提示
        prompt = self._build_execution_prompt(chain, question)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.2
            )
        
        result = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return self._clean_output(result)
    
    def _build_execution_prompt(self, chain: Dict, question: str) -> str:
        """
        构建执行提示
        """
        prompt = f"问题：{question}\n\n"
        prompt += "请按以下详细步骤进行推理：\n\n"
        
        def add_steps(steps: List[Dict], indent: int = 0):
            nonlocal prompt
            prefix = "  " * indent
            for i, step in enumerate(steps):
                prompt += f"{prefix}{i+1}. {step.get('name', '')}\n"
                if 'children' in step:
                    add_steps(step['children'], indent + 1)
        
        if 'steps' in chain:
            add_steps(chain['steps'])
        
        prompt += "\n请逐步执行上述推理，给出详细过程和最终答案：\n"
        
        return prompt
    
    def _parse_logic_chain(self, response: str) -> Dict:
        """解析模型生成的逻辑链"""
        chain = {'steps': []}
        
        lines = response.split('\n')
        current_step = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 匹配步骤
            if re.match(r'^步骤\d+', line):
                if current_step:
                    chain['steps'].append(current_step)
                current_step = {
                    'name': re.sub(r'^步骤\d+[：:]\s*', '', line),
                    'children': []
                }
            # 匹配子步骤
            elif current_step and re.match(r'^[-•]\s*', line):
                substep = re.sub(r'^[-•]\s*', '', line)
                current_step['children'].append({'name': substep})
        
        if current_step:
            chain['steps'].append(current_step)
        
        return chain
    
    def _calculate_density(self, chain: Dict) -> int:
        """计算逻辑链稠密度"""
        def count_nodes(steps: List[Dict]) -> int:
            count = 0
            for step in steps:
                count += 1
                if 'children' in step:
                    count += count_nodes(step['children'])
            return count
        
        return count_nodes(chain.get('steps', []))
    
    def _store_to_memory(self, chain: Dict, result: str):
        """存储到海马体记忆"""
        if self.hippocampus:
            # 创建记忆特征
            # 这里简化实现
            pass
    
    def _clean_output(self, text: str) -> str:
        """清理输出"""
        # 移除重复行
        lines = text.split('\n')
        seen = set()
        cleaned = []
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                cleaned.append(line)
        
        result = '\n'.join(cleaned)
        
        if len(result) > 500:
            result = result[:500] + "..."
        
        return result
    
    def _calculator_tool(self, expression: str) -> float:
        """计算器工具"""
        try:
            # 安全计算
            return eval(expression, {"__builtins__": {}}, {})
        except:
            return 0.0
    
    def _extractor_tool(self, text: str) -> Dict:
        """数值提取工具"""
        numbers = re.findall(r'\d+\.?\d*', text)
        return {'numbers': [float(n) for n in numbers]}
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'system': {
                'total_queries': self.stats['total_queries'],
                'total_time': self.stats['total_time'],
                'avg_density': self.stats['avg_density']
            },
            'fractal_engine': {
                'max_depth': self.fractal_engine.max_depth,
                'hidden_size': self.fractal_engine.hidden_size
            }
        }


def create_dense_logic_brain(model_path: str, device: str = "cpu") -> DenseLogicBrainAI:
    """创建稠密逻辑链类脑AI"""
    return DenseLogicBrainAI(model_path=model_path, device=device)
