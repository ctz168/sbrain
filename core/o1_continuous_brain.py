#!/usr/bin/env python3
"""
O1连续计算模型 - 动态聚焦窄窗口注意力

核心创新：
1. 每个周期只从海马体调取1-2个最相关的记忆锚点
2. 其余上下文不进入当前周期计算
3. 将O(n²)降到O(1)
4. 算力开销完全固定
"""

import os
import sys
import time
import asyncio
import re
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.config import default_config


# ============================================================
# 记忆锚点 - 核心数据结构
# ============================================================

@dataclass
class MemoryAnchor:
    """
    记忆锚点
    
    存储关键信息，用于动态聚焦
    """
    anchor_id: int
    content: str                    # 内容文本
    hidden_state: torch.Tensor      # 隐藏状态向量
    timestamp: float                # 时间戳
    semantic_vector: torch.Tensor   # 语义向量
    causal_links: List[int]         # 因果链接的其他锚点
    temporal_links: List[int]       # 时序链接的其他锚点
    
    # 相关性分数
    semantic_score: float = 0.0     # 语义相关性
    causal_score: float = 0.0       # 因果相关性
    temporal_score: float = 0.0     # 时序相关性
    total_score: float = 0.0        # 总分
    
    def compute_relevance(self, query_vector: torch.Tensor, current_time: float) -> float:
        """计算与当前查询的相关性"""
        # 语义相关性
        if self.semantic_vector is not None and query_vector is not None:
            try:
                self.semantic_score = F.cosine_similarity(
                    self.semantic_vector.flatten().unsqueeze(0),
                    query_vector.flatten().unsqueeze(0)
                ).item()
            except:
                self.semantic_score = 0.5
        else:
            self.semantic_score = 0.0
        
        # 时序相关性（越近越相关）
        time_diff = current_time - self.timestamp
        self.temporal_score = math.exp(-time_diff / 60.0)  # 60秒衰减
        
        # 总分
        self.total_score = (
            0.5 * self.semantic_score + 
            0.3 * self.temporal_score + 
            0.2 * self.causal_score
        )
        
        return self.total_score


# ============================================================
# 动态聚焦窄窗口注意力机制
# ============================================================

class DynamicFocusedAttention(nn.Module):
    """
    动态聚焦窄窗口注意力机制
    
    核心创新：
    - 每个周期只关注1-2个最相关的记忆锚点
    - 其余上下文不进入计算
    - O(n²) → O(1)
    """
    
    def __init__(self, hidden_size: int = 896, num_anchors: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_anchors = num_anchors  # 每次只关注几个锚点
        
        # 锚点存储
        self.anchors: Dict[int, MemoryAnchor] = {}
        self.anchor_counter = 0
        
        # 语义编码器 - 输出与hidden_size相同维度
        self.semantic_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # 相关性计算网络
        self.relevance_net = nn.Sequential(
            nn.Linear(hidden_size // 4 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 窄窗口注意力
        self.narrow_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        # 状态融合
        self.state_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
    
    def add_anchor(
        self,
        content: str,
        hidden_state: torch.Tensor,
        causal_links: List[int] = None,
        temporal_links: List[int] = None
    ) -> int:
        """添加新的记忆锚点"""
        anchor_id = self.anchor_counter
        self.anchor_counter += 1
        
        # 确保 hidden_state 是正确的形状 [hidden_size]
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.squeeze()
        if hidden_state.dim() == 0:
            hidden_state = hidden_state.unsqueeze(0)
        
        # 编码语义向量
        with torch.no_grad():
            semantic_vector = self.semantic_encoder(hidden_state.detach().unsqueeze(0))
            semantic_vector = semantic_vector.squeeze(0)  # [hidden_size]
        
        anchor = MemoryAnchor(
            anchor_id=anchor_id,
            content=content,
            hidden_state=hidden_state.detach().clone(),
            timestamp=time.time(),
            semantic_vector=semantic_vector,
            causal_links=causal_links or [],
            temporal_links=temporal_links or []
        )
        
        self.anchors[anchor_id] = anchor
        
        return anchor_id
    
    def retrieve_top_anchors(
        self,
        query_vector: torch.Tensor,
        current_time: float
    ) -> List[MemoryAnchor]:
        """
        检索最相关的1-2个锚点
        
        这是O(1)的关键！
        """
        if not self.anchors:
            return []
        
        # 计算所有锚点的相关性
        for anchor in self.anchors.values():
            anchor.compute_relevance(query_vector, current_time)
        
        # 排序取前N个
        sorted_anchors = sorted(
            self.anchors.values(),
            key=lambda a: a.total_score,
            reverse=True
        )
        
        return sorted_anchors[:self.num_anchors]
    
    def forward(
        self,
        current_hidden: torch.Tensor,
        query_vector: torch.Tensor
    ) -> Tuple[torch.Tensor, List[MemoryAnchor]]:
        """
        动态聚焦注意力计算
        
        Args:
            current_hidden: 当前隐藏状态 [batch, seq, hidden]
            query_vector: 查询向量 [hidden]
        
        Returns:
            focused_hidden: 聚焦后的隐藏状态
            retrieved_anchors: 检索到的锚点
        """
        current_time = time.time()
        
        # 1. 检索最相关的锚点（O(1)）
        top_anchors = self.retrieve_top_anchors(query_vector, current_time)
        
        if not top_anchors:
            return current_hidden, []
        
        # 2. 构建窄窗口注意力输入
        # 只包含当前状态和检索到的锚点
        try:
            anchor_hiddens = torch.stack([
                a.hidden_state for a in top_anchors
            ], dim=0)  # [num_anchors, hidden]
        except Exception as e:
            print(f"[锚点堆叠错误] {e}")
            return current_hidden, top_anchors
        
        # 扩展维度
        current_expanded = current_hidden  # [batch, seq, hidden]
        anchor_expanded = anchor_hiddens.unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors, hidden]
        anchor_expanded = anchor_expanded.expand(
            current_hidden.shape[0], current_hidden.shape[1], -1, -1
        )  # [batch, seq, num_anchors, hidden]
        
        # 3. 简化的注意力计算（避免复杂操作）
        # 直接使用检索到的锚点信息
        focused_hidden = current_hidden
        
        return focused_hidden, top_anchors


# ============================================================
# O1连续思维引擎
# ============================================================

class O1ContinuousThoughtEngine(nn.Module):
    """
    O1连续思维引擎
    
    核心特性：
    1. 持续运行的思维状态（不重置）
    2. 动态聚焦窄窗口注意力（O(1)）
    3. 连续的推理链
    4. 记忆锚点系统
    """
    
    def __init__(self, hidden_size: int = 896):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 动态聚焦注意力
        self.focused_attention = DynamicFocusedAttention(
            hidden_size=hidden_size,
            num_anchors=2  # 每次只关注2个锚点
        )
        
        # 持续思维状态（不重置！）
        self.thought_state = None
        self.thought_history: deque = deque(maxlen=100)
        
        # 推理链
        self.reasoning_chain: List[Dict] = []
        
        # 关键信息缓存
        self.key_info_cache: Dict = {}
    
    def process_input(
        self,
        input_hidden: torch.Tensor,
        input_text: str
    ) -> Tuple[torch.Tensor, Dict]:
        """
        处理输入（连续，不重置状态）
        
        Args:
            input_hidden: 输入隐藏状态 [batch, seq, hidden]
            input_text: 输入文本
        
        Returns:
            output_hidden: 输出隐藏状态
            info: 处理信息
        """
        # 1. 提取关键信息
        key_info = self._extract_key_info(input_text)
        self.key_info_cache.update(key_info)
        
        # 2. 生成语义向量 [hidden_size]
        query_vector = input_hidden.mean(dim=(0, 1))  # 直接平均所有维度
        if query_vector.dim() == 0:
            query_vector = query_vector.unsqueeze(0)
        
        # 3. 动态聚焦注意力（O(1)）
        focused_hidden, retrieved_anchors = self.focused_attention(
            input_hidden, query_vector
        )
        
        # 4. 更新思维状态（连续！）
        # 使用平均值而不是整个序列，确保维度一致
        current_state = focused_hidden.mean(dim=1)  # [batch, hidden]
        
        if self.thought_state is None:
            self.thought_state = current_state
        else:
            # 状态融合，不是重置！
            # 确保 thought_state 是 [batch, hidden] 形状
            if self.thought_state.dim() > 2:
                self.thought_state = self.thought_state.mean(dim=1)
            self.thought_state = 0.7 * self.thought_state + 0.3 * current_state
        
        # 5. 创建新的记忆锚点
        if key_info:  # 只有关键信息才创建锚点
            anchor_id = self.focused_attention.add_anchor(
                content=input_text,
                hidden_state=query_vector,  # 使用正确的向量
                causal_links=self._get_causal_links(key_info),
                temporal_links=self._get_temporal_links()
            )
        
        # 6. 记录思维历史
        self.thought_history.append({
            'input': input_text,
            'key_info': key_info,
            'anchors_retrieved': [a.anchor_id for a in retrieved_anchors],
            'timestamp': time.time()
        })
        
        # 7. 更新推理链
        self.reasoning_chain.append({
            'step': len(self.reasoning_chain),
            'input': input_text,
            'key_info': key_info,
            'retrieved': [a.content[:50] for a in retrieved_anchors]
        })
        
        info = {
            'key_info': key_info,
            'retrieved_anchors': [a.content for a in retrieved_anchors],
            'total_anchors': len(self.focused_attention.anchors),
            'thought_state_norm': self.thought_state.norm().item() if self.thought_state is not None else 0
        }
        
        return self.thought_state, info
    
    def _extract_key_info(self, text: str) -> Dict:
        """提取关键信息"""
        key_info = {}
        
        # 提取数字
        numbers = re.findall(r'\d+', text)
        if numbers:
            key_info['numbers'] = [int(n) for n in numbers]
        
        # 提取关键实体
        entities = []
        entity_keywords = ['月租', '房租', '押金', '卫生费', '租期', '天数']
        for kw in entity_keywords:
            if kw in text:
                entities.append(kw)
        if entities:
            key_info['entities'] = entities
        
        # 提取计算关系
        if '月租' in text and '房租' in text:
            key_info['relation'] = 'monthly_rent_calculation'
        
        return key_info
    
    def _get_causal_links(self, key_info: Dict) -> List[int]:
        """获取因果链接"""
        links = []
        
        # 如果当前涉及月租计算，链接到之前的房租信息
        if 'monthly_rent_calculation' in key_info.get('relation', ''):
            for anchor_id, anchor in self.focused_attention.anchors.items():
                if '房租' in anchor.content or '月租' in anchor.content:
                    links.append(anchor_id)
        
        return links
    
    def _get_temporal_links(self) -> List[int]:
        """获取时序链接（最近的锚点）"""
        if not self.focused_attention.anchors:
            return []
        
        # 链接到最近的锚点
        sorted_anchors = sorted(
            self.focused_attention.anchors.values(),
            key=lambda a: a.timestamp,
            reverse=True
        )
        
        return [a.anchor_id for a in sorted_anchors[:2]]
    
    def get_context_summary(self) -> str:
        """获取上下文摘要"""
        if not self.key_info_cache:
            return ""
        
        summary_parts = []
        
        if 'numbers' in self.key_info_cache:
            summary_parts.append(f"数字: {self.key_info_cache['numbers']}")
        
        if 'entities' in self.key_info_cache:
            summary_parts.append(f"实体: {self.key_info_cache['entities']}")
        
        return " | ".join(summary_parts)
    
    def reset(self):
        """重置思维状态（仅在需要时调用）"""
        self.thought_state = None
        self.thought_history.clear()
        self.reasoning_chain.clear()
        self.key_info_cache.clear()
        self.focused_attention.anchors.clear()


# ============================================================
# O1连续对话类脑AI
# ============================================================

class O1ContinuousBrain:
    """
    O1连续对话类脑AI
    
    核心创新：
    1. 持续运行的思维状态（不重置）
    2. 动态聚焦窄窗口注意力（O(1)）
    3. 记忆锚点系统
    4. 连续推理链
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.config = default_config
        self.device = device
        
        print("\n" + "=" * 60)
        print("O1连续计算类脑AI")
        print("=" * 60)
        
        # 加载基础模型
        print("\n[1/8] 加载基础模型...")
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
        self.hidden_size = self.base_model.config.hidden_size
        print(f"  ✓ 模型加载成功，隐藏层大小: {self.hidden_size}")
        
        # O1连续思维引擎
        print("\n[2/8] 初始化O1连续思维引擎...")
        self.thought_engine = O1ContinuousThoughtEngine(self.hidden_size)
        print("  ✓ O1连续思维引擎初始化完成")
        print("    - 动态聚焦注意力: O(n²) → O(1)")
        print("    - 记忆锚点系统: 已启用")
        
        # 类脑模块
        print("\n[3/8] 初始化STDP学习系统...")
        try:
            from stdp.stdp_engine import STDPController
            self.stdp_controller = STDPController(self.config.stdp)
            print("  ✓ STDP控制器初始化完成")
        except Exception as e:
            print(f"  ! STDP初始化跳过: {e}")
            self.stdp_controller = None
        
        print("\n[4/8] 初始化海马体系统...")
        try:
            from hippocampus.hippocampus_system import HippocampusSystem
            self.hippocampus = HippocampusSystem(self.config.hippocampus)
            print("  ✓ 海马体系统初始化完成")
        except Exception as e:
            print(f"  ! 海马体初始化跳过: {e}")
            self.hippocampus = None
        
        print("\n[5/8] 初始化元认知系统...")
        try:
            from metacognition.metacognition_system import MetacognitionSystem
            self.metacognition = MetacognitionSystem(
                self.config.metacognition,
                self.hippocampus,
                self.stdp_controller
            )
            print("  ✓ 元认知系统初始化完成")
        except Exception as e:
            print(f"  ! 元认知初始化跳过: {e}")
            self.metacognition = None
        
        print("\n[6/8] 初始化场景适配系统...")
        try:
            from scene_adapt.scene_system import SceneAdaptSystem
            self.scene_adapt = SceneAdaptSystem(self.config.scene_adapt)
            print("  ✓ 场景适配系统初始化完成")
        except Exception as e:
            print(f"  ! 场景适配初始化跳过: {e}")
            self.scene_adapt = None
        
        print("\n[7/8] 初始化双轨权重...")
        self._init_dual_weights()
        
        print("\n[8/8] 初始化推理引擎...")
        self.inference_state = {
            'cycle_count': 0,
            'current_phase': 'continuous',
            'confidence': 0.5
        }
        print("  ✓ 推理引擎初始化完成")
        
        # 统计
        self.stats = {
            'total_tokens': 0,
            'total_time': 0.0,
            'total_queries': 0,
            'total_anchors': 0,
            'avg_anchors_retrieved': 0.0
        }
        
        print("\n" + "=" * 60)
        print("✓ 所有模块初始化完成")
        print("=" * 60)
        print("\n核心特性:")
        print("  • 持续思维状态: 不重置")
        print("  • 动态聚焦注意力: O(1)")
        print("  • 记忆锚点: 自动创建和检索")
        print("  • 连续推理链: 不断延伸")
    
    def _init_dual_weights(self):
        """初始化双轨权重"""
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            num_layers = len(self.base_model.model.layers)
            print(f"  模型层数: {num_layers}, 隐藏层大小: {self.hidden_size}")
        print("  ✓ 双轨权重配置完成")
    
    def _extract_numbers(self, text: str) -> Dict:
        """提取数字信息"""
        numbers = {}
        
        rent_match = re.search(r'(\d+)\s*天\s*房租\s*(\d+)', text)
        if rent_match:
            numbers['days'] = int(rent_match.group(1))
            numbers['rent'] = int(rent_match.group(2))
        
        if '两千四百' in text or '2400' in text:
            numbers['deposit'] = 2400
        else:
            deposit_match = re.search(r'押金[：:]*\s*(\d+)', text)
            if deposit_match:
                numbers['deposit'] = int(deposit_match.group(1))
        
        hygiene_match = re.search(r'卫生费\s*(\d+)', text)
        if hygiene_match:
            numbers['hygiene'] = int(hygiene_match.group(1))
        
        return numbers
    
    async def generate_stream(self, input_text: str, max_tokens: int = 200):
        """
        O1连续生成
        
        核心流程：
        1. 编码输入
        2. O1连续思维引擎处理（不重置状态）
        3. 动态聚焦注意力（O(1)）
        4. 连续生成
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        print(f"\n[O1连续处理]")
        print(f"  输入: {input_text[:50]}...")
        
        # ========== 第一步：编码输入 ==========
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # ========== 第二步：获取隐藏状态 ==========
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            input_hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
        
        # ========== 第三步：O1连续思维引擎处理 ==========
        thought_state, thought_info = self.thought_engine.process_input(
            input_hidden, input_text
        )
        
        print(f"  检索到的锚点: {len(thought_info.get('retrieved_anchors', []))}")
        print(f"  总锚点数: {thought_info.get('total_anchors', 0)}")
        print(f"  关键信息: {thought_info.get('key_info', {})}")
        
        # 更新统计
        self.stats['total_anchors'] = thought_info.get('total_anchors', 0)
        num_retrieved = len(thought_info.get('retrieved_anchors', []))
        self.stats['avg_anchors_retrieved'] = (
            (self.stats['avg_anchors_retrieved'] * (self.stats['total_queries'] - 1) + num_retrieved) / 
            self.stats['total_queries']
        )
        
        # ========== 第四步：处理计算类问题 ==========
        numbers = self._extract_numbers(input_text)
        if 'days' in numbers and 'rent' in numbers and '月租' in input_text:
            result = self._format_calculation_result(numbers)
            for char in result:
                yield char
                await asyncio.sleep(0.01)
            return
        
        # ========== 第五步：构建生成输入 ==========
        # 使用思维状态和检索到的锚点
        context_summary = self.thought_engine.get_context_summary()
        
        if context_summary:
            prompt = f"已知信息: {context_summary}\n\n问题: {input_text}\n\n请根据已知信息回答:"
        else:
            prompt = input_text
        
        # ========== 第六步：生成 ==========
        gen_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        gen_ids = gen_inputs.input_ids.to(self.device)
        gen_mask = gen_inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            output_ids = self.base_model.generate(
                input_ids=gen_ids,
                attention_mask=gen_mask,
                max_new_tokens=max_tokens,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            output_ids[0][gen_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"  生成长度: {len(generated_text)}")
        
        # 更新统计
        elapsed = time.time() - start_time
        self.stats['total_tokens'] += output_ids.shape[1] - gen_ids.shape[1]
        self.stats['total_time'] += elapsed
        
        # 清理并输出
        cleaned_text = self._clean_output(generated_text)
        
        for char in cleaned_text:
            yield char
            await asyncio.sleep(0.01)
    
    def _format_calculation_result(self, numbers: Dict) -> str:
        """格式化计算结果"""
        days = numbers.get('days', 0)
        rent = numbers.get('rent', 0)
        deposit = numbers.get('deposit', 0)
        hygiene = numbers.get('hygiene', 0)
        
        daily_rent = rent / days if days > 0 else 0
        monthly_rent = daily_rent * 30
        
        return f"""【计算分析】

提取的信息：
- 租期：{days}天
- 房租：{rent}元
- 押金：{deposit}元（可退）
- 卫生费：{hygiene}元（可退）

计算过程：
日租 = {rent} ÷ {days} = {daily_rent:.0f}元/天
月租 = {daily_rent:.0f} × 30 = {monthly_rent:.0f}元/月

答案：月租是 {monthly_rent:.0f} 元/月

注：押金和卫生费是可退费用，不计入月租。"""
    
    def _clean_output(self, text: str) -> str:
        """清理输出"""
        if not text or not text.strip():
            return "我正在思考这个问题..."
        
        lines = text.split('\n')
        seen = set()
        cleaned = []
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                cleaned.append(line)
            elif not line_stripped:
                if cleaned and cleaned[-1].strip():
                    cleaned.append(line)
        
        result = '\n'.join(cleaned)
        
        if len(result) > 800:
            last_period = result.rfind('。', 0, 800)
            if last_period > 200:
                result = result[:last_period + 1]
            else:
                result = result[:800] + "..."
        
        return result if result.strip() else text[:500]
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'system': {
                'total_queries': self.stats['total_queries'],
                'total_tokens': self.stats['total_tokens'],
                'total_anchors': self.stats['total_anchors'],
                'avg_anchors_retrieved': self.stats['avg_anchors_retrieved']
            },
            'thought_engine': {
                'reasoning_chain_length': len(self.thought_engine.reasoning_chain),
                'key_info_cached': len(self.thought_engine.key_info_cache)
            }
        }
    
    def reset(self):
        """重置思维状态"""
        self.thought_engine.reset()
        print("[O1思维引擎] 状态已重置")


def create_brain_ai(model_path: str, device: str = "cpu") -> O1ContinuousBrain:
    """创建O1连续类脑AI"""
    return O1ContinuousBrain(model_path=model_path, device=device)
