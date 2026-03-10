"""
模块1：双轨权重原生改造
基于Qwen3.5-0.8B原生Transformer架构

核心功能：
- 权重按9:1拆分为静态分支与动态分支
- 静态分支：90%冻结，继承官方预训练权重
- 动态分支：10%可更新，仅通过STDP规则
- 前向融合：9:1加权融合输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
import math


class DualWeightLinear(nn.Module):
    """
    双轨权重线性层
    
    将权重按列维度9:1拆分：
    - 静态分支(90%)：冻结，继承预训练权重
    - 动态分支(10%)：可更新，STDP驱动
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        static_weight: Optional[torch.Tensor] = None,
        bias: bool = True,
        static_ratio: float = 0.9,
        dynamic_init_std: float = 0.02
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.static_ratio = static_ratio
        self.dynamic_ratio = 1.0 - static_ratio
        
        # 计算分割点
        self.static_out_features = int(out_features * static_ratio)
        self.dynamic_out_features = out_features - self.static_out_features
        
        # 静态分支权重（冻结）
        if static_weight is not None:
            # 从预训练权重中提取静态部分
            self.register_buffer(
                'static_weight', 
                static_weight[:self.static_out_features, :].clone()
            )
        else:
            # 初始化静态权重
            self.register_buffer(
                'static_weight',
                torch.empty(self.static_out_features, in_features)
            )
            nn.init.kaiming_uniform_(self.static_weight, a=math.sqrt(5))
        
        # 动态分支权重（可训练，但仅通过STDP更新）
        self.dynamic_weight = nn.Parameter(
            torch.randn(self.dynamic_out_features, in_features) * dynamic_init_std
        )
        
        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # STDP追踪
        self.pre_activation = None
        self.post_activation = None
        self.stdp_update_count = 0
        
        # 冻结静态权重
        self.static_weight.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：静态分支与动态分支加权融合
        
        Args:
            x: 输入张量 [batch, seq_len, in_features]
        
        Returns:
            output: 融合输出 [batch, seq_len, out_features]
        """
        # 保存前激活用于STDP
        self.pre_activation = x.detach()
        
        # 静态分支计算
        static_output = F.linear(x, self.static_weight, None)
        
        # 动态分支计算
        dynamic_output = F.linear(x, self.dynamic_weight, None)
        
        # 9:1加权融合
        # 静态分支权重为0.9，动态分支权重为0.1
        output = torch.cat([
            static_output * self.static_ratio,
            dynamic_output * self.dynamic_ratio
        ], dim=-1)
        
        # 保存后激活用于STDP
        self.post_activation = output.detach()
        
        # 添加偏置
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def apply_stdp(
        self,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor,
        confidence: float,
        alpha: float = 0.005,
        beta: float = 0.01,
        t_pre_threshold: float = 0.5,
        t_post_threshold: float = 0.5
    ):
        """
        应用STDP规则更新动态权重
        
        LTP: Δw = α * pre * post * (t_post - threshold) * confidence
        LTD: Δw = -β * pre * post * (t_pre - threshold) * (1 - confidence)
        """
        with torch.no_grad():
            # 计算时序信号
            pre_mean = pre_activation.mean(dim=-1, keepdim=True)
            post_mean = post_activation.mean(dim=-1, keepdim=True)
            
            # 归一化激活
            pre_norm = (pre_activation - pre_mean) / (pre_activation.std() + 1e-8)
            post_norm = (post_activation - post_mean) / (post_activation.std() + 1e-8)
            
            # 计算STDP更新
            if confidence >= 0.5:
                # 长期增强LTP
                t_post = torch.sigmoid(post_norm.mean())
                delta_w = alpha * pre_norm.mean() * post_norm.mean() * (t_post - t_post_threshold) * confidence
            else:
                # 长期抑制LTD
                t_pre = torch.sigmoid(pre_norm.mean())
                delta_w = -beta * pre_norm.mean() * post_norm.mean() * (t_pre - t_pre_threshold) * (1 - confidence)
            
            # 应用更新
            self.dynamic_weight.data += delta_w
            
            # 权重裁剪
            self.dynamic_weight.data.clamp_(-0.1, 0.1)
            
            self.stdp_update_count += 1
    
    def get_dynamic_weights(self) -> torch.Tensor:
        """获取动态权重"""
        return self.dynamic_weight.data.clone()
    
    def set_dynamic_weights(self, weights: torch.Tensor):
        """设置动态权重"""
        with torch.no_grad():
            self.dynamic_weight.data = weights.clone()


class DualWeightAttention(nn.Module):
    """
    双轨权重注意力层
    
    对Q/K/V投影和输出投影均实现双轨拆分
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        static_q: Optional[torch.Tensor] = None,
        static_k: Optional[torch.Tensor] = None,
        static_v: Optional[torch.Tensor] = None,
        static_o: Optional[torch.Tensor] = None,
        static_ratio: float = 0.9
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.static_ratio = static_ratio
        
        # 双轨Q/K/V投影
        self.q_proj = DualWeightLinear(
            hidden_size, hidden_size, 
            static_weight=static_q,
            static_ratio=static_ratio
        )
        self.k_proj = DualWeightLinear(
            hidden_size, hidden_size,
            static_weight=static_k,
            static_ratio=static_ratio
        )
        self.v_proj = DualWeightLinear(
            hidden_size, hidden_size,
            static_weight=static_v,
            static_ratio=static_ratio
        )
        self.o_proj = DualWeightLinear(
            hidden_size, hidden_size,
            static_weight=static_o,
            static_ratio=static_ratio
        )
        
        # 注意力特征输出（用于海马体输入）
        self.attention_features = None
        self.temporal_features = None
        self.semantic_features = None
        
        # 海马体门控信号
        self.hippocampus_gate = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_anchors: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: 可选的注意力掩码
            memory_anchors: 海马体输出的记忆锚点门控信号
            output_attentions: 是否输出注意力权重
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 应用海马体门控（如果有）
        if memory_anchors is not None:
            self.hippocampus_gate = memory_anchors
            # 门控公式：attention_weight = origin * (0.7 + 0.3 * memory_confidence)
            gate_factor = 0.7 + 0.3 * memory_anchors.mean()
        else:
            gate_factor = 1.0
        
        # Q/K/V投影
        query = self.q_proj(hidden_states) * gate_factor
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 重塑为多头形式
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 注意力输出
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # 输出投影
        output = self.o_proj(context)
        
        # 提取特征用于海马体
        self.attention_features = attention_probs.mean(dim=1).detach()  # [batch, seq_len, head_dim]
        self.temporal_features = query.mean(dim=1).detach()
        self.semantic_features = context.detach()
        
        if output_attentions:
            return output, attention_probs
        return output, None
    
    def apply_stdp_to_all(
        self,
        confidence: float,
        alpha: float = 0.005,
        beta: float = 0.01
    ):
        """对所有投影层应用STDP"""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            if proj.pre_activation is not None and proj.post_activation is not None:
                proj.apply_stdp(
                    proj.pre_activation,
                    proj.post_activation,
                    confidence,
                    alpha, beta
                )
    
    def get_all_dynamic_weights(self) -> Dict[str, torch.Tensor]:
        """获取所有动态权重"""
        return {
            'q_dynamic': self.q_proj.get_dynamic_weights(),
            'k_dynamic': self.k_proj.get_dynamic_weights(),
            'v_dynamic': self.v_proj.get_dynamic_weights(),
            'o_dynamic': self.o_proj.get_dynamic_weights()
        }


class DualWeightFFN(nn.Module):
    """
    双轨权重前馈网络
    
    对gate_proj、up_proj、down_proj实现双轨拆分
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        static_gate: Optional[torch.Tensor] = None,
        static_up: Optional[torch.Tensor] = None,
        static_down: Optional[torch.Tensor] = None,
        static_ratio: float = 0.9
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # 双轨FFN层
        self.gate_proj = DualWeightLinear(
            hidden_size, intermediate_size,
            static_weight=static_gate,
            static_ratio=static_ratio
        )
        self.up_proj = DualWeightLinear(
            hidden_size, intermediate_size,
            static_weight=static_up,
            static_ratio=static_ratio
        )
        self.down_proj = DualWeightLinear(
            intermediate_size, hidden_size,
            static_weight=static_down,
            static_ratio=static_ratio
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：SwiGLU激活"""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        output = self.down_proj(gate * up)
        return output
    
    def apply_stdp_to_all(
        self,
        confidence: float,
        alpha: float = 0.005,
        beta: float = 0.01
    ):
        """对所有FFN层应用STDP"""
        for proj in [self.gate_proj, self.up_proj, self.down_proj]:
            if proj.pre_activation is not None and proj.post_activation is not None:
                proj.apply_stdp(
                    proj.pre_activation,
                    proj.post_activation,
                    confidence,
                    alpha, beta
                )
    
    def get_all_dynamic_weights(self) -> Dict[str, torch.Tensor]:
        """获取所有动态权重"""
        return {
            'gate_dynamic': self.gate_proj.get_dynamic_weights(),
            'up_dynamic': self.up_proj.get_dynamic_weights(),
            'down_dynamic': self.down_proj.get_dynamic_weights()
        }


class DualWeightTransformerLayer(nn.Module):
    """
    完整的双轨权重Transformer层
    
    包含：
    - 双轨注意力层
    - 双轨FFN层
    - 层归一化
    - 残差连接
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        layer_norm_eps: float = 1e-6,
        static_ratio: float = 0.9
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 双轨注意力
        self.self_attn = DualWeightAttention(
            hidden_size, num_attention_heads,
            static_ratio=static_ratio
        )
        
        # 双轨FFN
        self.mlp = DualWeightFFN(
            hidden_size, intermediate_size,
            static_ratio=static_ratio
        )
        
        # 层归一化
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_anchors: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播"""
        # 自注意力 + 残差
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, attn_weights = self.self_attn(
            hidden_states, attention_mask, memory_anchors, output_attentions
        )
        hidden_states = residual + attn_output
        
        # FFN + 残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states, attn_weights
    
    def apply_stdp_to_all(
        self,
        confidence: float,
        alpha: float = 0.005,
        beta: float = 0.01
    ):
        """应用STDP到所有层"""
        self.self_attn.apply_stdp_to_all(confidence, alpha, beta)
        self.mlp.apply_stdp_to_all(confidence, alpha, beta)
    
    def get_all_dynamic_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """获取所有动态权重"""
        return {
            'attention': self.self_attn.get_all_dynamic_weights(),
            'ffn': self.mlp.get_all_dynamic_weights()
        }


# ==================== 角色适配接口 ====================

class RoleAdapter:
    """
    角色适配器
    
    支持通过固定提示词模板实现角色切换：
    - 提案者(Proposer)
    - 验证者(Validator)
    - 辩论者(Debater)
    - 裁判(Judge)
    """
    
    ROLE_TEMPLATES = {
        'proposer': """你是一个创意提案者。你的任务是：
1. 提出多个可能的解决方案或观点
2. 每个方案都要有清晰的逻辑支撑
3. 保持开放思维，不急于下结论
请针对以下问题提出你的方案：""",
        
        'validator': """你是一个严谨的验证者。你的任务是：
1. 检查每个方案的事实准确性（0-10分）
2. 评估逻辑连贯性（0-10分）
3. 指出潜在的问题和漏洞
请对以下方案进行验证：""",
        
        'debater': """你是一个批判性辩论者。你的任务是：
1. 对方案进行交叉验证
2. 补充逻辑漏洞
3. 剔除错误内容
请对以下方案进行辩论：""",
        
        'judge': """你是一个公正的裁判。你的任务是：
1. 综合评估所有方案
2. 给出最终打分（0-10分）
3. 选择最优方案并说明理由
请对以下方案做出裁决："""
    }
    
    @classmethod
    def get_role_prompt(cls, role: str) -> str:
        """获取角色提示词"""
        return cls.ROLE_TEMPLATES.get(role, "")
    
    @classmethod
    def format_input(cls, role: str, content: str) -> str:
        """格式化输入"""
        role_prompt = cls.get_role_prompt(role)
        return f"{role_prompt}\n\n{content}"
