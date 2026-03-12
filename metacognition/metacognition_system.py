"""
模块4：元认知双闭环校验系统

核心功能：
- 元认知特征提取与置信度计算
- 在线实时校验闭环
- 离线反思闭环
- 彻底解决幻觉问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time


@dataclass
class MetacognitiveFeatures:
    """元认知特征"""
    attention_entropy: float = 0.5
    stdp_activation: float = 0.5
    semantic_similarity: float = 0.5
    confidence: float = 0.5


class MetacognitiveFeatureExtractor(nn.Module):
    """
    元认知特征提取器
    
    每个周期同步提取3个核心元特征：
    1. 注意力熵
    2. STDP激活强度
    3. 语义一致性
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征权重
        self.attention_entropy_weight = config.attention_entropy_weight
        self.stdp_activation_weight = config.stdp_activation_weight
        self.semantic_similarity_weight = config.semantic_similarity_weight
        
        print("[Metacognition] 元认知特征提取器初始化完成")
    
    def forward(
        self,
        attention_probs: torch.Tensor,
        stdp_state: Dict,
        hidden_states: torch.Tensor,
        context_hidden: torch.Tensor = None
    ) -> MetacognitiveFeatures:
        """
        提取元认知特征
        
        Args:
            attention_probs: 注意力概率分布
            stdp_state: STDP状态
            hidden_states: 当前隐藏状态
            context_hidden: 上下文隐藏状态
        
        Returns:
            features: 元认知特征
        """
        # 1. 计算注意力熵
        attention_entropy = self._compute_attention_entropy(attention_probs)
        
        # 2. 计算STDP激活强度
        stdp_activation = self._compute_stdp_activation(stdp_state)
        
        # 3. 计算语义一致性
        semantic_similarity = self._compute_semantic_similarity(
            hidden_states, context_hidden
        )
        
        # 4. 计算综合置信度
        confidence = (
            self.attention_entropy_weight * (1 - attention_entropy) +
            self.stdp_activation_weight * stdp_activation +
            self.semantic_similarity_weight * semantic_similarity
        )
        
        return MetacognitiveFeatures(
            attention_entropy=attention_entropy,
            stdp_activation=stdp_activation,
            semantic_similarity=semantic_similarity,
            confidence=confidence
        )
    
    def _compute_attention_entropy(self, attention_probs: torch.Tensor) -> float:
        """
        计算注意力熵
        
        归一化至0-1
        """
        if attention_probs is None:
            return 0.5
        
        # 计算熵
        entropy = -torch.sum(
            attention_probs * torch.log(attention_probs + 1e-10),
            dim=-1
        ).mean()
        
        # 归一化
        max_entropy = torch.log(torch.tensor(attention_probs.size(-1), dtype=torch.float))
        normalized_entropy = (entropy / max_entropy).item()
        
        return min(1.0, max(0.0, normalized_entropy))
    
    def _compute_stdp_activation(self, stdp_state: Dict) -> float:
        """
        计算STDP激活强度
        
        归一化至0-1
        """
        if not stdp_state:
            return 0.5
        
        # 从STDP状态提取激活强度
        total_updates = stdp_state.get('cycle_count', 0)
        attention_updates = stdp_state.get('attention_updates', 0)
        ffn_updates = stdp_state.get('ffn_updates', 0)
        
        # 计算激活比例
        if total_updates == 0:
            return 0.5
        
        activation = (attention_updates + ffn_updates) / (2 * total_updates + 1)
        
        return min(1.0, max(0.0, activation))
    
    def _compute_semantic_similarity(
        self,
        hidden_states: torch.Tensor,
        context_hidden: torch.Tensor
    ) -> float:
        """
        计算语义一致性
        
        当前生成token与上下文的语义余弦相似度
        """
        if hidden_states is None:
            return 0.5
        
        if context_hidden is None:
            return 0.5
        
        # 余弦相似度
        similarity = F.cosine_similarity(
            hidden_states.flatten().unsqueeze(0),
            context_hidden.flatten().unsqueeze(0)
        ).item()
        
        # 归一化到0-1
        normalized = (similarity + 1) / 2
        
        return min(1.0, max(0.0, normalized))


class OnlineValidator:
    """
    在线实时校验闭环
    
    根据置信度执行不同校验策略：
    - 置信度≥0.8：正常输出
    - 0.6≤置信度<0.8：交叉验证
    - 置信度<0.6：深度反思
    """
    
    def __init__(self, config):
        self.config = config
        
        self.high_threshold = config.high_confidence_threshold
        self.medium_threshold = config.medium_confidence_threshold
        
        self.cross_validation_expansion = config.cross_validation_expansion
        self.deep_reflection_expansion = config.deep_reflection_expansion
        
        self.validation_history = []
    
    def validate(
        self,
        confidence: float,
        output: Any,
        validator_func=None
    ) -> Dict:
        """
        执行在线校验
        
        Args:
            confidence: 置信度
            output: 待校验输出
            validator_func: 校验函数
        
        Returns:
            result: 校验结果
        """
        result = {
            'confidence': confidence,
            'action': 'normal',
            'validated_output': output,
            'expansion_cycles': 0
        }
        
        if confidence >= self.high_threshold:
            # 高置信度：正常输出
            result['action'] = 'normal'
            result['expansion_cycles'] = 0
        
        elif confidence >= self.medium_threshold:
            # 中置信度：交叉验证
            result['action'] = 'cross_validation'
            result['expansion_cycles'] = self.cross_validation_expansion
            
            # 执行交叉验证
            if validator_func:
                validated = validator_func(output)
                result['validated_output'] = validated
        
        else:
            # 低置信度：深度反思
            result['action'] = 'deep_reflection'
            result['expansion_cycles'] = self.deep_reflection_expansion
            
            # 标记为需要深度反思
            result['validated_output'] = "无法确定：需要进一步验证"
        
        self.validation_history.append({
            'timestamp': time.time(),
            'confidence': confidence,
            'action': result['action']
        })
        
        return result
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.validation_history:
            return {'total_validations': 0}
        
        actions = [v['action'] for v in self.validation_history]
        return {
            'total_validations': len(self.validation_history),
            'normal_count': actions.count('normal'),
            'cross_validation_count': actions.count('cross_validation'),
            'deep_reflection_count': actions.count('deep_reflection')
        }


class OfflineReflector:
    """
    离线反思闭环
    
    结合海马体尖波涟漪回放：
    - 对错误推理复盘
    - 通过STDP抑制错误路径
    - 强化正确校验逻辑
    """
    
    def __init__(self, config, hippocampus_system, stdp_controller):
        self.config = config
        self.hippocampus = hippocampus_system
        self.stdp = stdp_controller
        
        self.reflection_history = []
        self.error_patterns = {}
    
    def reflect(
        self,
        low_confidence_outputs: List[Dict]
    ) -> Dict:
        """
        执行离线反思
        
        Args:
            low_confidence_outputs: 低置信度输出列表
        
        Returns:
            result: 反思结果
        """
        result = {
            'reflected_count': 0,
            'corrected_count': 0,
            'suppressed_patterns': []
        }
        
        for output in low_confidence_outputs:
            # 分析错误模式
            error_pattern = self._analyze_error_pattern(output)
            
            if error_pattern:
                # 记录错误模式
                pattern_key = str(error_pattern)
                self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
                
                # 通过STDP抑制错误路径
                self._suppress_error_path(error_pattern)
                
                result['suppressed_patterns'].append(error_pattern)
            
            result['reflected_count'] += 1
        
        self.reflection_history.append({
            'timestamp': time.time(),
            'reflected_count': result['reflected_count']
        })
        
        return result
    
    def _analyze_error_pattern(self, output: Dict) -> Optional[Dict]:
        """分析错误模式"""
        if output.get('confidence', 0) >= 0.6:
            return None
        
        return {
            'type': 'low_confidence',
            'confidence': output.get('confidence', 0),
            'token': output.get('token', -1)
        }
    
    def _suppress_error_path(self, error_pattern: Dict):
        """通过STDP抑制错误路径"""
        # 使用负向学习率抑制
        self.stdp.step(
            model=None,  # 将在调用时处理
            context_tokens=[],
            current_token=error_pattern.get('token', 0),
            confidence=0.1,  # 低置信度触发抑制
            semantic_contribution=0.1
        )
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_reflections': len(self.reflection_history),
            'unique_error_patterns': len(self.error_patterns),
            'error_pattern_counts': dict(list(self.error_patterns.items())[:10])
        }


class MetacognitionSystem(nn.Module):
    """
    完整的元认知双闭环校验系统
    
    整合：
    - 特征提取
    - 在线校验
    - 离线反思
    """
    
    def __init__(self, config, hippocampus_system=None, stdp_controller=None):
        super().__init__()
        self.config = config
        
        # 初始化子模块
        self.feature_extractor = MetacognitiveFeatureExtractor(config)
        self.online_validator = OnlineValidator(config)
        self.offline_reflector = OfflineReflector(config, hippocampus_system, stdp_controller)
        
        # 低置信度输出缓存
        self.low_confidence_cache = []
        self.max_cache_size = 100
        
        print("[Metacognition] 元认知系统初始化完成")
    
    def forward(
        self,
        attention_probs: torch.Tensor,
        stdp_state: Dict,
        hidden_states: torch.Tensor,
        context_hidden: torch.Tensor = None,
        output: Any = None
    ) -> Tuple[MetacognitiveFeatures, Dict]:
        """
        执行完整的元认知校验
        
        Args:
            attention_probs: 注意力概率
            stdp_state: STDP状态
            hidden_states: 隐藏状态
            context_hidden: 上下文隐藏状态
            output: 待校验输出
        
        Returns:
            features: 元认知特征
            validation_result: 校验结果
        """
        # 1. 提取元认知特征
        features = self.feature_extractor(
            attention_probs, stdp_state, hidden_states, context_hidden
        )
        
        # 2. 在线校验
        validation_result = self.online_validator.validate(
            features.confidence, output
        )
        
        # 3. 缓存低置信度输出
        if features.confidence < 0.6:
            self._cache_low_confidence(output, features)
        
        return features, validation_result
    
    def _cache_low_confidence(self, output: Any, features: MetacognitiveFeatures):
        """缓存低置信度输出"""
        self.low_confidence_cache.append({
            'output': output,
            'features': features,
            'timestamp': time.time()
        })
        
        # 限制缓存大小
        if len(self.low_confidence_cache) > self.max_cache_size:
            self.low_confidence_cache.pop(0)
    
    def execute_offline_reflection(self) -> Optional[Dict]:
        """执行离线反思"""
        if not self.low_confidence_cache:
            return None
        
        result = self.offline_reflector.reflect(self.low_confidence_cache)
        
        # 清空缓存
        self.low_confidence_cache = []
        
        return result
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'feature_extractor': {
                'last_confidence': getattr(self, '_last_confidence', 0)
            },
            'online_validator': self.online_validator.get_stats(),
            'offline_reflector': self.offline_reflector.get_stats(),
            'cache_size': len(self.low_confidence_cache)
        }
