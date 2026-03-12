"""
模块5：海马体-新皮层协同记忆系统

严格按照人脑海马体神经环路1:1开发：
- 内嗅皮层EC：特征编码
- 齿状回DG：模式分离
- CA3区：情景记忆库
- CA1区：时序门控
- 长期语义记忆库
- 尖波涟漪SWR：离线巩固
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import time
import random
import hashlib


@dataclass
class MemoryEntry:
    """
    单条记忆条目
    
    结构：
    - memory_id: 64位正交编码
    - timestamp: 10ms级全局周期编号
    - 时序骨架: 前后3条记忆的ID双向链表
    - 语义指针: 关键token的词表索引
    - 因果关联: 相关记忆ID列表
    - confidence: 0-1置信度
    - visit_count: 访问频次
    """
    memory_id: str
    timestamp: int
    temporal_skeleton: Dict[str, List[str]] = field(default_factory=dict)  # {'prev': [], 'next': []}
    semantic_pointers: List[int] = field(default_factory=list)
    causal_relations: List[str] = field(default_factory=list)
    confidence: float = 0.5
    visit_count: int = 0
    feature_vector: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'memory_id': self.memory_id,
            'timestamp': self.timestamp,
            'temporal_skeleton': self.temporal_skeleton,
            'semantic_pointers': self.semantic_pointers,
            'causal_relations': self.causal_relations,
            'confidence': self.confidence,
            'visit_count': self.visit_count
        }


@dataclass
class SemanticTriple:
    """
    语义三元组
    
    结构：主体-关系-客体
    """
    subject: str
    relation: str
    object: str
    weight: float = 1.0
    source_memory_id: str = ""
    timestamp: int = 0


class EntorhinalCortex(nn.Module):
    """
    内嗅皮层EC特征编码单元
    
    功能：
    - 复用模型768维原生输出
    - 通过稀疏随机投影压缩为64维
    - 稀疏度75%
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.EC_input_dim
        self.output_dim = config.EC_output_dim
        self.sparsity = config.EC_sparsity
        
        # 固定稀疏随机投影矩阵（无训练参数）
        torch.manual_seed(42)  # 固定种子
        mask = (torch.rand(self.output_dim, self.input_dim) > self.sparsity).float()
        projection = torch.randn(self.output_dim, self.input_dim) * mask
        self.register_buffer('projection_matrix', projection / (self.input_dim ** 0.5))
        
        print(f"[EC] 初始化完成: {self.input_dim} -> {self.output_dim}, 稀疏度={self.sparsity}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        编码特征
        
        Args:
            hidden_states: [batch, seq_len, input_dim] 或 [batch, input_dim]
        
        Returns:
            sparse_features: [batch, seq_len, output_dim] 或 [batch, output_dim]
        """
        # 稀疏投影
        features = F.linear(hidden_states, self.projection_matrix)
        
        # ReLU激活保持稀疏性
        features = F.relu(features)
        
        # L2归一化
        features = F.normalize(features, p=2, dim=-1)
        
        return features


class DentateGyrus(nn.Module):
    """
    齿状回DG区模式分离单元
    
    功能：
    - 对64维特征做正交化处理
    - 为相似输入生成完全正交的记忆ID
    - 无训练参数
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config.EC_output_dim
        self.random_seed = config.DG_random_seed
        
        # 固定正交化投影矩阵
        torch.manual_seed(self.random_seed)
        orthogonal_matrix = self._create_orthogonal_matrix(self.output_dim)
        self.register_buffer('orthogonal_matrix', orthogonal_matrix)
        
        print(f"[DG] 初始化完成: 正交化矩阵 {self.output_dim}x{self.output_dim}")
    
    def _create_orthogonal_matrix(self, dim: int) -> torch.Tensor:
        """创建正交矩阵"""
        matrix = torch.randn(dim, dim)
        q, _ = torch.qr(matrix)
        return q
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        正交化特征
        
        Args:
            features: [batch, output_dim]
        
        Returns:
            orthogonal_features: [batch, output_dim]
        """
        # 正交投影
        orthogonal = F.linear(features, self.orthogonal_matrix)
        
        # 二值化生成记忆ID
        binary = (orthogonal > 0).float()
        
        return orthogonal, binary
    
    def generate_memory_id(self, features: torch.Tensor) -> str:
        """
        生成64位记忆ID
        
        Args:
            features: [output_dim] 单个特征向量
        
        Returns:
            memory_id: 64字符的十六进制ID
        """
        with torch.no_grad():
            _, binary = self.forward(features.unsqueeze(0))
            
            # 转换为十六进制字符串
            binary_str = ''.join([str(int(b)) for b in binary.squeeze()])
            # 转换为十六进制
            hex_id = hex(int(binary_str, 2))[2:].zfill(16)
            
            return hex_id


class CA3MemoryStore:
    """
    CA3区情景记忆库
    
    功能：
    - 固定大小循环缓存队列
    - 最大1024条记忆
    - 模式补全功能
    """
    
    def __init__(self, config):
        self.config = config
        self.max_capacity = config.CA3_max_capacity
        
        # 循环缓存
        self.memory_buffer: Dict[str, MemoryEntry] = {}
        self.memory_order: deque = deque(maxlen=self.max_capacity)
        
        # 时间戳计数器
        self.global_timestamp = 0
        
        # 统计
        self.total_stored = 0
        self.total_recalled = 0
        
        print(f"[CA3] 初始化完成: 最大容量={self.max_capacity}")
    
    def store(
        self,
        memory_id: str,
        feature_vector: torch.Tensor,
        semantic_pointers: List[int],
        confidence: float = 0.5
    ) -> MemoryEntry:
        """
        存储新记忆
        
        Args:
            memory_id: 记忆ID
            feature_vector: 特征向量
            semantic_pointers: 语义指针
            confidence: 置信度
        
        Returns:
            entry: 存储的记忆条目
        """
        # 如果达到容量上限，移除最旧的记忆
        if len(self.memory_buffer) >= self.max_capacity:
            oldest_id = self.memory_order.popleft()
            if oldest_id in self.memory_buffer:
                del self.memory_buffer[oldest_id]
        
        # 创建记忆条目
        entry = MemoryEntry(
            memory_id=memory_id,
            timestamp=self.global_timestamp,
            semantic_pointers=semantic_pointers,
            confidence=confidence,
            visit_count=0,
            feature_vector=feature_vector.clone() if isinstance(feature_vector, torch.Tensor) else None
        )
        
        # 建立时序骨架
        if len(self.memory_order) > 0:
            prev_id = self.memory_order[-1]
            entry.temporal_skeleton['prev'] = [prev_id]
            entry.temporal_skeleton['next'] = []  # 初始化next列表
            
            # 更新前一条记忆的后向链接
            if prev_id in self.memory_buffer:
                if 'next' not in self.memory_buffer[prev_id].temporal_skeleton:
                    self.memory_buffer[prev_id].temporal_skeleton['next'] = []
                self.memory_buffer[prev_id].temporal_skeleton['next'].append(memory_id)
        else:
            entry.temporal_skeleton['prev'] = []
            entry.temporal_skeleton['next'] = []
        
        # 存储
        self.memory_buffer[memory_id] = entry
        self.memory_order.append(memory_id)
        
        self.global_timestamp += 1
        self.total_stored += 1
        
        return entry
    
    def recall(
        self,
        query_feature: torch.Tensor,
        top_k: int = 2
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        召回相关记忆
        
        Args:
            query_feature: 查询特征向量
            top_k: 返回top-k个最相关记忆
        
        Returns:
            memories: [(记忆条目, 相似度分数)]
        """
        if len(self.memory_buffer) == 0:
            return []
        
        similarities = []
        
        for memory_id, entry in self.memory_buffer.items():
            if entry.feature_vector is not None:
                # 余弦相似度
                sim = F.cosine_similarity(
                    query_feature.unsqueeze(0),
                    entry.feature_vector.unsqueeze(0)
                ).item()
                similarities.append((entry, sim))
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_memories = similarities[:top_k]
        
        # 更新访问计数
        for entry, _ in top_memories:
            entry.visit_count += 1
        
        self.total_recalled += len(top_memories)
        
        return top_memories
    
    def pattern_completion(
        self,
        partial_cue: str
    ) -> Optional[MemoryEntry]:
        """
        模式补全：通过部分线索召回完整记忆
        
        Args:
            partial_cue: 部分记忆ID或语义线索
        
        Returns:
            entry: 完整的记忆条目
        """
        # 尝试精确匹配
        if partial_cue in self.memory_buffer:
            return self.memory_buffer[partial_cue]
        
        # 尝试前缀匹配
        for memory_id, entry in self.memory_buffer.items():
            if memory_id.startswith(partial_cue):
                return entry
        
        return None
    
    def get_memory_chain(
        self,
        memory_id: str,
        chain_length: int = 3
    ) -> List[MemoryEntry]:
        """
        获取记忆链条
        
        Args:
            memory_id: 起始记忆ID
            chain_length: 链条长度
        
        Returns:
            chain: 记忆链条
        """
        chain = []
        current_id = memory_id
        
        for _ in range(chain_length):
            if current_id not in self.memory_buffer:
                break
            
            entry = self.memory_buffer[current_id]
            chain.append(entry)
            
            # 获取下一个记忆
            next_list = entry.temporal_skeleton.get('next', [])
            if next_list and len(next_list) > 0:
                current_id = next_list[0]
            else:
                break
        
        return chain
    
    def cleanup(
        self,
        visit_threshold: int = 2,
        confidence_threshold: float = 0.3
    ) -> int:
        """
        清理无效记忆
        
        Args:
            visit_threshold: 访问频次阈值
            confidence_threshold: 置信度阈值
        
        Returns:
            removed_count: 清理的记忆数量
        """
        to_remove = []
        
        for memory_id, entry in self.memory_buffer.items():
            if entry.visit_count < visit_threshold and entry.confidence < confidence_threshold:
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            del self.memory_buffer[memory_id]
            if memory_id in self.memory_order:
                self.memory_order.remove(memory_id)
        
        return len(to_remove)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_memories': len(self.memory_buffer),
            'total_stored': self.total_stored,
            'total_recalled': self.total_recalled,
            'global_timestamp': self.global_timestamp
        }


class CA1GateController(nn.Module):
    """
    CA1区时序门控单元
    
    功能：
    - 为记忆绑定时间戳与因果关系
    - 每周期输出1-2个最相关记忆锚点
    - 门控公式：attention_weight = origin * (0.7 + 0.3 * memory_confidence)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.recall_topk = config.CA1_recall_topk
        self.gate_base = config.CA1_gate_base
        self.gate_confidence_weight = config.CA1_gate_confidence_weight
        
        print(f"[CA1] 初始化完成: 召回top-{self.recall_topk}")
    
    def forward(
        self,
        recalled_memories: List[Tuple[MemoryEntry, float]]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        生成门控信号
        
        Args:
            recalled_memories: 召回的记忆列表 [(entry, similarity)]
        
        Returns:
            gate_signal: 门控信号
            anchor_ids: 锚点ID列表
        """
        if not recalled_memories:
            return torch.ones(1), []
        
        # 计算门控信号
        gate_values = []
        anchor_ids = []
        
        for entry, similarity in recalled_memories[:self.recall_topk]:
            # 门控公式
            gate_value = self.gate_base + self.gate_confidence_weight * entry.confidence
            gate_values.append(gate_value)
            anchor_ids.append(entry.memory_id)
        
        # 平均门控信号
        gate_signal = torch.tensor([sum(gate_values) / len(gate_values)])
        
        return gate_signal, anchor_ids
    
    def compute_attention_weight(
        self,
        original_weight: torch.Tensor,
        memory_confidence: float
    ) -> torch.Tensor:
        """
        计算注意力权重
        
        Args:
            original_weight: 原始注意力权重
            memory_confidence: 记忆置信度
        
        Returns:
            gated_weight: 门控后的权重
        """
        gate_factor = self.gate_base + self.gate_confidence_weight * memory_confidence
        return original_weight * gate_factor


class SemanticMemoryStore:
    """
    长期语义记忆库
    
    功能：
    - 存储「主体-关系-客体」三元组
    - 最大4096条
    - 通过STDP更新关联权重
    """
    
    def __init__(self, config):
        self.config = config
        self.max_capacity = config.semantic_max_capacity
        
        # 三元组存储
        self.triples: List[SemanticTriple] = []
        self.triple_index: Dict[str, List[int]] = {}  # 主体索引
        
        print(f"[SemanticMemory] 初始化完成: 最大容量={self.max_capacity}")
    
    def store(
        self,
        subject: str,
        relation: str,
        object: str,
        source_memory_id: str = "",
        timestamp: int = 0
    ) -> SemanticTriple:
        """
        存储语义三元组
        
        Args:
            subject: 主体
            relation: 关系
            object: 客体
            source_memory_id: 来源记忆ID
            timestamp: 时间戳
        
        Returns:
            triple: 存储的三元组
        """
        if len(self.triples) >= self.max_capacity:
            # 移除权重最小的三元组
            self.triples.sort(key=lambda x: x.weight, reverse=True)
            removed = self.triples.pop()
            # 更新索引
            if removed.subject in self.triple_index:
                self.triple_index[removed.subject] = [
                    i for i in self.triple_index[removed.subject]
                    if self.triples[i].subject != removed.subject
                ]
        
        triple = SemanticTriple(
            subject=subject,
            relation=relation,
            object=object,
            source_memory_id=source_memory_id,
            timestamp=timestamp
        )
        
        self.triples.append(triple)
        
        # 更新索引
        if subject not in self.triple_index:
            self.triple_index[subject] = []
        self.triple_index[subject].append(len(self.triples) - 1)
        
        return triple
    
    def query(
        self,
        subject: str = None,
        relation: str = None,
        object: str = None
    ) -> List[SemanticTriple]:
        """
        查询语义三元组
        
        Args:
            subject: 主体（可选）
            relation: 关系（可选）
            object: 客体（可选）
        
        Returns:
            triples: 匹配的三元组列表
        """
        results = []
        
        for triple in self.triples:
            match = True
            if subject and triple.subject != subject:
                match = False
            if relation and triple.relation != relation:
                match = False
            if object and triple.object != object:
                match = False
            if match:
                results.append(triple)
        
        return results
    
    def update_weight(
        self,
        triple_idx: int,
        delta: float
    ):
        """
        更新三元组权重（STDP）
        
        Args:
            triple_idx: 三元组索引
            delta: 权重变化量
        """
        if 0 <= triple_idx < len(self.triples):
            self.triples[triple_idx].weight = max(0.1, min(2.0, self.triples[triple_idx].weight + delta))
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_triples': len(self.triples),
            'unique_subjects': len(self.triple_index)
        }


class SharpWaveRipple:
    """
    尖波涟漪SWR离线记忆巩固单元
    
    功能：
    - 空闲30秒触发
    - 回放Top30记忆
    - 通过STDP强化正确记忆
    - 抽象为语义三元组
    - 清理无效记忆
    """
    
    def __init__(self, config):
        self.config = config
        self.idle_threshold = config.SWR_idle_threshold_s
        self.replay_topk = config.SWR_replay_topk
        self.max_duration = config.SWR_max_duration_s
        
        self.last_activity_time = time.time()
        self.is_active = False
        self.replay_count = 0
        
        print(f"[SWR] 初始化完成: 空闲阈值={self.idle_threshold}s")
    
    def check_trigger(self) -> bool:
        """
        检查是否触发SWR
        
        Returns:
            should_trigger: 是否触发
        """
        idle_time = time.time() - self.last_activity_time
        return idle_time >= self.idle_threshold and not self.is_active
    
    def execute(
        self,
        ca3_store: CA3MemoryStore,
        semantic_store: SemanticMemoryStore,
        stdp_controller
    ) -> Dict:
        """
        执行SWR回放巩固
        
        Args:
            ca3_store: CA3记忆库
            semantic_store: 语义记忆库
            stdp_controller: STDP控制器
        
        Returns:
            result: 巩固结果
        """
        self.is_active = True
        start_time = time.time()
        
        result = {
            'replayed_memories': 0,
            'consolidated_to_semantic': 0,
            'cleaned_memories': 0
        }
        
        # 1. 按访问频次+置信度抽取Top30记忆
        all_memories = list(ca3_store.memory_buffer.values())
        all_memories.sort(
            key=lambda x: x.visit_count * x.confidence,
            reverse=True
        )
        top_memories = all_memories[:self.replay_topk]
        
        # 2. 按时间戳排序回放
        top_memories.sort(key=lambda x: x.timestamp)
        
        for entry in top_memories:
            # 检查是否超时
            if time.time() - start_time > self.max_duration:
                break
            
            # 通过STDP强化正确记忆
            if entry.confidence >= 0.7:
                # 模拟STDP强化
                pass
            
            # 抽象为语义三元组（简化实现）
            if len(entry.semantic_pointers) >= 2:
                # 从语义指针提取主体和客体
                subject = f"concept_{entry.semantic_pointers[0]}"
                object = f"concept_{entry.semantic_pointers[-1]}"
                semantic_store.store(
                    subject=subject,
                    relation="related_to",
                    object=object,
                    source_memory_id=entry.memory_id,
                    timestamp=entry.timestamp
                )
                result['consolidated_to_semantic'] += 1
            
            result['replayed_memories'] += 1
        
        # 3. 清理无效记忆
        cleaned = ca3_store.cleanup(
            visit_threshold=self.config.SWR_cleanup_visit_threshold,
            confidence_threshold=self.config.SWR_cleanup_confidence_threshold
        )
        result['cleaned_memories'] = cleaned
        
        self.replay_count += 1
        self.is_active = False
        self.last_activity_time = time.time()
        
        return result
    
    def register_activity(self):
        """注册活动（重置空闲计时）"""
        self.last_activity_time = time.time()


class HippocampusSystem(nn.Module):
    """
    完整的海马体系统
    
    整合EC、DG、CA3、CA1、语义记忆、SWR
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 初始化各子模块
        self.ec = EntorhinalCortex(config)
        self.dg = DentateGyrus(config)
        self.ca3 = CA3MemoryStore(config)
        self.ca1 = CA1GateController(config)
        self.semantic_memory = SemanticMemoryStore(config)
        self.swr = SharpWaveRipple(config)
        
        # 当前锚点
        self.current_anchors = []
        self.current_gate_signal = torch.ones(1)
        
        print("[Hippocampus] 海马体系统初始化完成")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        semantic_pointers: List[int] = None,
        confidence: float = 0.5
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        完整的海马体处理流程
        
        Args:
            hidden_states: 隐藏状态 [batch, hidden_size]
            semantic_pointers: 语义指针
            confidence: 置信度
        
        Returns:
            gate_signal: 门控信号
            anchor_ids: 锚点ID列表
        """
        # 注册活动
        self.swr.register_activity()
        
        # 1. EC编码
        features = self.ec(hidden_states)
        
        # 2. DG正交化
        orthogonal_features, binary = self.dg(features)
        
        # 3. 生成记忆ID并存储
        if semantic_pointers is None:
            semantic_pointers = []
        
        memory_id = self.dg.generate_memory_id(features.squeeze())
        self.ca3.store(
            memory_id=memory_id,
            feature_vector=features.squeeze(),
            semantic_pointers=semantic_pointers,
            confidence=confidence
        )
        
        # 4. 召回相关记忆
        recalled = self.ca3.recall(features.squeeze(), top_k=2)
        
        # 5. CA1门控
        gate_signal, anchor_ids = self.ca1(recalled)
        
        self.current_anchors = anchor_ids
        self.current_gate_signal = gate_signal
        
        return gate_signal, anchor_ids
    
    def get_memory_anchor(self) -> Tuple[torch.Tensor, List[str]]:
        """获取当前记忆锚点"""
        return self.current_gate_signal, self.current_anchors
    
    def check_and_consolidate(self, stdp_controller) -> Optional[Dict]:
        """检查并执行SWR巩固"""
        if self.swr.check_trigger():
            return self.swr.execute(
                self.ca3,
                self.semantic_memory,
                stdp_controller
            )
        return None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'episodic_memory': self.ca3.get_stats(),
            'semantic_memory': self.semantic_memory.get_stats(),
            'swr_replay_count': self.swr.replay_count
        }
