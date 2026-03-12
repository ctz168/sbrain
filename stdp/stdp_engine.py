"""
模块3：全链路精细化STDP时序可塑性学习系统

核心功能：
- 100%替代反向传播
- 每个10ms周期自动执行
- 全链路STDP更新覆盖
- 动态学习率适配
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time


@dataclass
class STDPState:
    """STDP状态追踪"""
    pre_activation: Optional[torch.Tensor] = None
    post_activation: Optional[torch.Tensor] = None
    timestamp: float = 0.0
    confidence: float = 0.5
    update_count: int = 0


class STDPUpdateRule:
    """
    STDP核心更新规则
    
    LTP公式：Δw = α * pre * post * (t_post - threshold) * confidence
    LTD公式：Δw = -β * pre * post * (t_pre - threshold) * (1 - confidence)
    """
    
    def __init__(
        self,
        alpha: float = 0.005,  # 正向学习率
        beta: float = 0.01,    # 负向学习率
        weight_min: float = -0.1,
        weight_max: float = 0.1,
        t_pre_threshold: float = 0.5,
        t_post_threshold: float = 0.5
    ):
        self.alpha = alpha
        self.beta = beta
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.t_pre_threshold = t_pre_threshold
        self.t_post_threshold = t_post_threshold
        
        # 动态学习率适配
        self.high_confidence_boost = 1.5
        self.low_confidence_boost = 2.0
    
    def compute_update(
        self,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor,
        confidence: float,
        current_time: float = None
    ) -> torch.Tensor:
        """
        计算STDP权重更新量
        
        Args:
            pre_activation: 前神经元激活
            post_activation: 后神经元激活
            confidence: 置信度 [0, 1]
            current_time: 当前时间戳
        
        Returns:
            delta_w: 权重更新量
        """
        # 动态学习率适配
        effective_alpha = self.alpha
        effective_beta = self.beta
        
        if confidence >= 0.9:
            effective_alpha *= self.high_confidence_boost
        elif confidence < 0.6:
            effective_beta *= self.low_confidence_boost
        
        # 归一化激活值
        pre_norm = self._normalize_activation(pre_activation)
        post_norm = self._normalize_activation(post_activation)
        
        # 计算时序信号
        t_pre = torch.sigmoid(pre_norm.mean())
        t_post = torch.sigmoid(post_norm.mean())
        
        # 计算相关性
        correlation = self._compute_correlation(pre_norm, post_norm)
        
        # STDP更新
        if confidence >= 0.5:
            # 长期增强LTP
            delta_w = effective_alpha * correlation * (t_post - self.t_post_threshold) * confidence
        else:
            # 长期抑制LTD
            delta_w = -effective_beta * correlation * (t_pre - self.t_pre_threshold) * (1 - confidence)
        
        return delta_w
    
    def _normalize_activation(self, activation: torch.Tensor) -> torch.Tensor:
        """归一化激活值"""
        if activation is None or activation.numel() == 0:
            return torch.zeros(1)
        
        mean = activation.mean()
        std = activation.std() + 1e-8
        return (activation - mean) / std
    
    def _compute_correlation(
        self,
        pre: torch.Tensor,
        post: torch.Tensor
    ) -> torch.Tensor:
        """计算前后激活的相关性"""
        # 简化的相关性计算
        if pre.shape != post.shape:
            # 如果形状不匹配，使用均值
            return pre.mean() * post.mean()
        
        return (pre * post).mean()
    
    def apply_update(
        self,
        weight: torch.Tensor,
        delta_w: torch.Tensor
    ) -> torch.Tensor:
        """应用权重更新并裁剪"""
        with torch.no_grad():
            weight.add_(delta_w)
            weight.clamp_(self.weight_min, self.weight_max)
        return weight


class AttentionSTDP:
    """
    注意力层STDP更新器
    
    根据窄窗口内上下文与当前token的时序关联更新权重
    """
    
    def __init__(self, config):
        self.config = config
        self.rule = STDPUpdateRule(
            alpha=config.alpha_LTP,
            beta=config.beta_LTD,
            weight_min=config.weight_min,
            weight_max=config.weight_max
        )
        self.update_history = []
    
    def update(
        self,
        attention_layer,
        context_tokens: List[int],
        current_token: int,
        confidence: float,
        semantic_contribution: float = 0.5
    ):
        """
        更新注意力层动态权重
        
        Args:
            attention_layer: 双轨注意力层
            context_tokens: 窄窗口上下文token列表
            current_token: 当前token
            confidence: 置信度
            semantic_contribution: 语义贡献度
        """
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj = getattr(attention_layer, proj_name, None)
            if proj is None or not hasattr(proj, 'dynamic_weight'):
                continue
            
            if proj.pre_activation is None or proj.post_activation is None:
                continue
            
            # 计算STDP更新
            delta_w = self.rule.compute_update(
                proj.pre_activation,
                proj.post_activation,
                confidence * semantic_contribution
            )
            
            # 应用更新
            self.rule.apply_update(proj.dynamic_weight.data, delta_w)
        
        self.update_history.append({
            'timestamp': time.time(),
            'confidence': confidence,
            'context_size': len(context_tokens)
        })


class FFNSTDP:
    """
    FFN层STDP更新器
    
    对高频特征、专属术语、用户习惯增强权重
    """
    
    def __init__(self, config):
        self.config = config
        self.rule = STDPUpdateRule(
            alpha=config.alpha_LTP,
            beta=config.beta_LTD
        )
        self.feature_frequency = {}  # 特征频率追踪
    
    def update(
        self,
        ffn_layer,
        features: torch.Tensor,
        confidence: float,
        task_type: str = "general"
    ):
        """
        更新FFN层动态权重
        
        Args:
            ffn_layer: 双轨FFN层
            features: 当前特征
            confidence: 置信度
            task_type: 任务类型
        """
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(ffn_layer, proj_name, None)
            if proj is None or not hasattr(proj, 'dynamic_weight'):
                continue
            
            if proj.pre_activation is None or proj.post_activation is None:
                continue
            
            # 计算STDP更新
            delta_w = self.rule.compute_update(
                proj.pre_activation,
                proj.post_activation,
                confidence
            )
            
            # 应用更新
            self.rule.apply_update(proj.dynamic_weight.data, delta_w)
    
    def track_feature_frequency(self, feature_id: int):
        """追踪特征频率"""
        self.feature_frequency[feature_id] = self.feature_frequency.get(feature_id, 0) + 1


class HippocampusGateSTDP:
    """
    海马体门控STDP更新器
    
    对有正向贡献的记忆锚点增强权重
    """
    
    def __init__(self, config):
        self.config = config
        self.rule = STDPUpdateRule(
            alpha=config.alpha_LTP,
            beta=config.beta_LTD
        )
        self.anchor_contribution_history = {}
    
    def update(
        self,
        attention_layer,
        memory_anchor_id: str,
        anchor_contribution: float,
        confidence: float
    ):
        """
        更新海马体门控权重
        
        Args:
            attention_layer: 注意力层
            memory_anchor_id: 记忆锚点ID
            anchor_contribution: 锚点贡献度
            confidence: 置信度
        """
        # 记录锚点贡献历史
        if memory_anchor_id not in self.anchor_contribution_history:
            self.anchor_contribution_history[memory_anchor_id] = []
        self.anchor_contribution_history[memory_anchor_id].append(anchor_contribution)
        
        # 根据贡献度调整学习
        if anchor_contribution > 0.5:
            # 正向贡献，增强连接
            effective_confidence = confidence * anchor_contribution
        else:
            # 负向贡献，减弱连接
            effective_confidence = confidence * (1 - anchor_contribution)
        
        # 更新注意力层
        if hasattr(attention_layer, 'q_proj'):
            proj = attention_layer.q_proj
            if proj.pre_activation is not None and proj.post_activation is not None:
                delta_w = self.rule.compute_update(
                    proj.pre_activation,
                    proj.post_activation,
                    effective_confidence
                )
                self.rule.apply_update(proj.dynamic_weight.data, delta_w)


class SelfEvalSTDP:
    """
    自评判STDP更新器
    
    根据置信度打分、自校验结果更新权重
    """
    
    def __init__(self, config):
        self.config = config
        self.rule = STDPUpdateRule(
            alpha=config.alpha_LTP,
            beta=config.beta_LTD
        )
        self.evaluation_history = []
    
    def update(
        self,
        model_layers: List,
        path_scores: Dict[int, float],
        best_path: int
    ):
        """
        根据自评判结果更新权重
        
        Args:
            model_layers: 模型层列表
            path_scores: 各路径得分 {path_id: score}
            best_path: 最优路径ID
        """
        for path_id, score in path_scores.items():
            # 最优路径强化，其他路径抑制
            if path_id == best_path:
                confidence = score / 10.0  # 归一化到[0,1]
            else:
                confidence = 1.0 - (score / 10.0)
            
            # 更新所有层
            for layer in model_layers:
                if hasattr(layer, 'apply_stdp_to_all'):
                    layer.apply_stdp_to_all(
                        confidence,
                        self.config.alpha_LTP,
                        self.config.beta_LTD
                    )
        
        self.evaluation_history.append({
            'timestamp': time.time(),
            'path_scores': path_scores,
            'best_path': best_path
        })


class STDPController:
    """
    STDP总控制器
    
    协调所有STDP更新器的执行
    """
    
    def __init__(self, config):
        self.config = config
        
        # 初始化各更新器
        self.attention_stdp = AttentionSTDP(config)
        self.ffn_stdp = FFNSTDP(config)
        self.hippocampus_stdp = HippocampusGateSTDP(config)
        self.self_eval_stdp = SelfEvalSTDP(config)
        
        # 全局状态
        self.cycle_count = 0
        self.last_update_time = time.time()
        self.update_stats = {
            'attention_updates': 0,
            'ffn_updates': 0,
            'hippocampus_updates': 0,
            'self_eval_updates': 0
        }
    
    def step(
        self,
        model,
        context_tokens: List[int],
        current_token: int,
        confidence: float,
        memory_anchor_id: Optional[str] = None,
        anchor_contribution: float = 0.0,
        semantic_contribution: float = 0.5
    ):
        """
        执行一个STDP更新周期
        
        Args:
            model: 双轨权重模型
            context_tokens: 上下文token
            current_token: 当前token
            confidence: 置信度
            memory_anchor_id: 记忆锚点ID
            anchor_contribution: 锚点贡献度
            semantic_contribution: 语义贡献度
        """
        self.cycle_count += 1
        
        # 获取模型层
        layers = self._get_model_layers(model)
        
        # 1. 注意力层STDP更新
        if self.config.update_attention:
            for layer in layers:
                if hasattr(layer, 'self_attn'):
                    self.attention_stdp.update(
                        layer.self_attn,
                        context_tokens,
                        current_token,
                        confidence,
                        semantic_contribution
                    )
                    self.update_stats['attention_updates'] += 1
        
        # 2. FFN层STDP更新
        if self.config.update_ffn:
            for layer in layers:
                if hasattr(layer, 'mlp'):
                    self.ffn_stdp.update(
                        layer.mlp,
                        None,  # features
                        confidence
                    )
                    self.update_stats['ffn_updates'] += 1
        
        # 3. 海马体门控STDP更新
        if self.config.update_hippocampus_gate and memory_anchor_id:
            for layer in layers:
                if hasattr(layer, 'self_attn'):
                    self.hippocampus_stdp.update(
                        layer.self_attn,
                        memory_anchor_id,
                        anchor_contribution,
                        confidence
                    )
                    self.update_stats['hippocampus_updates'] += 1
        
        self.last_update_time = time.time()
    
    def apply_self_evaluation(
        self,
        model,
        path_scores: Dict[int, float],
        best_path: int
    ):
        """应用自评判STDP更新"""
        if self.config.update_self_eval:
            layers = self._get_model_layers(model)
            self.self_eval_stdp.update(layers, path_scores, best_path)
            self.update_stats['self_eval_updates'] += 1
    
    def _get_model_layers(self, model) -> List:
        """获取模型层列表"""
        if hasattr(model, 'layers'):
            return model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        return []
    
    def get_stats(self) -> Dict:
        """获取STDP统计信息"""
        return {
            'cycle_count': self.cycle_count,
            'last_update_time': self.last_update_time,
            **self.update_stats
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.cycle_count = 0
        self.update_stats = {
            'attention_updates': 0,
            'ffn_updates': 0,
            'hippocampus_updates': 0,
            'self_eval_updates': 0
        }
