#!/usr/bin/env python3
"""
逻辑自相似稠密补齐 (LSDC) 核心算法

数学模型：
1. 离散状态转移: S_n → S_{n+1}
2. 窄宽带补齐: f(S_n, Δt) → S_{n+1}
3. 自相似结构: [前提, 推演, 结论]

核心特性：
- 每个微观步的结构与宏观结构同构
- 强制逻辑稠密性
- 自动补齐跳过的步骤
"""

import os
import sys
import re
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
from enum import Enum

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsdc_engine.model_handler import ModelHandler, NarrowBandwidthConfig


class LogicPhase(Enum):
    """逻辑阶段 - 自相似结构的三位一体"""
    PREMISE = "前提"      # 前提：已知条件
    DERIVATION = "推演"   # 推演：逻辑过程
    CONCLUSION = "结论"   # 结论：结果


@dataclass
class LogicNode:
    """
    逻辑节点 S_n
    
    每个节点代表一个离散的逻辑状态
    """
    node_id: int
    phase: LogicPhase
    content: str
    premise: str = ""         # 前提
    derivation: str = ""      # 推演
    conclusion: str = ""      # 结论
    
    # 隐藏状态（用于状态连续性）
    hidden_state: Optional[torch.Tensor] = None
    
    # 逻辑稠密度
    density: float = 1.0
    
    def is_complete(self) -> bool:
        """检查节点是否完整（三位一体）"""
        return bool(self.premise and self.derivation and self.conclusion)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "phase": self.phase.value,
            "content": self.content,
            "premise": self.premise,
            "derivation": self.derivation,
            "conclusion": self.conclusion,
            "density": self.density
        }


@dataclass
class LogicChain:
    """
    逻辑链
    
    由多个逻辑节点组成的推理链
    """
    nodes: List[LogicNode] = field(default_factory=list)
    goal: str = ""
    
    def add_node(self, node: LogicNode):
        """添加节点"""
        self.nodes.append(node)
    
    def get_last_conclusion(self) -> Optional[str]:
        """获取最后一个结论"""
        for node in reversed(self.nodes):
            if node.conclusion:
                return node.conclusion
        return None
    
    def to_text(self) -> str:
        """转换为文本"""
        lines = []
        for node in self.nodes:
            if node.premise:
                lines.append(f"【{node.phase.value}】{node.premise}")
            if node.derivation:
                lines.append(f"  → {node.derivation}")
            if node.conclusion:
                lines.append(f"  ∴ {node.conclusion}")
        return "\n".join(lines)


class LogicProcessor:
    """
    逻辑处理器
    
    核心算法：逻辑自相似稠密补齐 (LSDC)
    """
    
    def __init__(
        self,
        model_handler: ModelHandler,
        max_iterations: int = 20,
        density_threshold: float = 0.5
    ):
        self.model = model_handler
        self.max_iterations = max_iterations
        self.density_threshold = density_threshold
        
        # 当前逻辑链
        self.current_chain: Optional[LogicChain] = None
        
        # 节点计数器
        self.node_counter = 0
    
    def process(
        self,
        goal: str,
        context: Optional[str] = None
    ) -> Generator[LogicNode, None, None]:
        """
        处理目标，生成逻辑链
        
        核心流程：
        1. 初始化逻辑链
        2. 循环生成微步
        3. 检查逻辑稠密性
        4. 补齐缺失步骤
        5. 更新状态
        
        Args:
            goal: 目标问题
            context: 上下文信息
        
        Yields:
            LogicNode: 每个生成的逻辑节点
        """
        # 初始化逻辑链
        self.current_chain = LogicChain(goal=goal)
        
        # 当前状态
        current_goal = goal
        previous_conclusion = context
        
        # 迭代生成
        for i in range(self.max_iterations):
            # 创建新节点
            node = self._create_node(current_goal, previous_conclusion)
            
            # 生成微步
            prompt = self.model.narrow_bandwidth_filter(
                current_goal, previous_conclusion
            )
            
            generated_text, hidden_state = self.model.generate_micro_step(prompt)
            
            # 解析生成内容
            node = self._parse_generated_text(node, generated_text)
            node.hidden_state = hidden_state
            
            # 检查逻辑稠密性
            if not self.model.is_logic_dense(generated_text):
                # 需要补齐
                node.density = 0.3
                yield node
                
                # 插入补齐步骤
                densify_node = self._densify(node, generated_text)
                yield densify_node
                previous_conclusion = densify_node.conclusion
            else:
                node.density = 1.0
                yield node
                previous_conclusion = node.conclusion
            
            # 添加到链
            self.current_chain.add_node(node)
            
            # 检查是否完成
            if self._is_goal_reached(node, goal):
                break
            
            # 更新目标
            current_goal = self._derive_next_goal(node, goal)
    
    def _create_node(
        self,
        goal: str,
        previous_conclusion: Optional[str]
    ) -> LogicNode:
        """创建新节点"""
        node = LogicNode(
            node_id=self.node_counter,
            phase=LogicPhase.DERIVATION,
            content="",
            premise=previous_conclusion or goal
        )
        self.node_counter += 1
        return node
    
    def _parse_generated_text(
        self,
        node: LogicNode,
        text: str
    ) -> LogicNode:
        """
        解析生成文本
        
        提取前提、推演、结论
        """
        # 移除<think...</think标签及其内容
        import re
        text = re.sub(r'<think.*?</think.*?>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think.*', '', text, flags=re.IGNORECASE)
        
        # 清理重复内容
        lines = text.strip().split('\n')
        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 跳过重复行
            if line == prev_line:
                continue
            # 跳过"详细推理过程"等无意义内容
            if '详细推理过程' in line or '推理过程' == line:
                continue
            # 跳过Thinking Process
            if 'Thinking Process' in line:
                continue
            cleaned_lines.append(line)
            prev_line = line
        
        text = ' '.join(cleaned_lines)
        
        # 尝试提取结构
        premise_match = re.search(r'已知[：:]\s*(.+?)(?=\n|问题|$)', text)
        derivation_match = re.search(r'(?:因为|由于|所以|因此|分析)[：:，]?\s*(.+?)(?=\n|结论|$)', text)
        conclusion_match = re.search(r'(?:结论|答案|结果|是)[：:]\s*(.+?)(?=\n|$)', text)
        
        if premise_match:
            node.premise = premise_match.group(1).strip()
        if derivation_match:
            node.derivation = derivation_match.group(1).strip()
        if conclusion_match:
            node.conclusion = conclusion_match.group(1).strip()
        
        # 如果没有明确结构，整体作为推演
        if not node.derivation and not node.conclusion:
            node.derivation = text.strip()[:150]
            # 尝试提取最后一句作为结论
            sentences = re.split(r'[。！？]', text)
            for s in reversed(sentences):
                s = s.strip()
                if s and len(s) > 2:
                    node.conclusion = s[:80]
                    break
        
        node.content = text[:300]
        return node
    
    def _densify(
        self,
        node: LogicNode,
        text: str
    ) -> LogicNode:
        """
        稠密化补齐
        
        当逻辑跨度过大时，插入补齐步骤
        """
        # 创建补齐节点
        densify_node = LogicNode(
            node_id=self.node_counter,
            phase=LogicPhase.DERIVATION,
            content="",
            premise=node.premise
        )
        self.node_counter += 1
        
        # 生成补齐内容
        densify_prompt = f"""解构上述步骤：
已知: {node.premise}
需要补充的推理: {text}
详细推理过程:"""
        
        densify_text, hidden_state = self.model.generate_micro_step(densify_prompt)
        
        densify_node = self._parse_generated_text(densify_node, densify_text)
        densify_node.hidden_state = hidden_state
        densify_node.density = 0.8
        
        return densify_node
    
    def _is_goal_reached(self, node: LogicNode, goal: str) -> bool:
        """检查是否达到目标"""
        # 简单实现：检查结论是否包含答案关键词
        if not node.conclusion:
            return False
        
        # 检查是否有明确的答案
        answer_patterns = [
            r'答案[是为]',
            r'结果[是为]',
            r'等于',
            r'\d+',  # 包含数字
        ]
        
        for pattern in answer_patterns:
            if re.search(pattern, node.conclusion):
                return True
        
        return False
    
    def _derive_next_goal(self, node: LogicNode, original_goal: str) -> str:
        """推导下一个目标"""
        if node.conclusion:
            return f"基于「{node.conclusion[:30]}」，继续推理"
        return original_goal
    
    def get_chain(self) -> Optional[LogicChain]:
        """获取当前逻辑链"""
        return self.current_chain


def create_logic_processor(
    model_path: str = "../models/Qwen3.5-0.8B",
    device: str = "cpu"
) -> LogicProcessor:
    """创建逻辑处理器"""
    model_handler = ModelHandler(model_path=model_path, device=device)
    return LogicProcessor(model_handler)
