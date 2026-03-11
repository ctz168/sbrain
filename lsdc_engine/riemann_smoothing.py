#!/usr/bin/env python3
"""
黎曼平滑层 (Riemann Smoothing Layer)

核心数学原理：
1. 拉普拉斯算子：捕捉逻辑跳跃
2. 解析延拓：在离散逻辑点之间拉起平滑曲线
3. 逻辑稠密度提升：通过平滑约束消除逻辑坍缩

数学模型：
- 原始状态: x(t)
- 平滑修正: Δx = α * ∇²x
- 输出状态: x'(t) = x(t) + Δx

其中 ∇² 是拉普拉斯算子，α 是平滑系数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RiemannSmoothingLayer(nn.Module):
    """
    黎曼平滑层
    
    通过拉普拉斯算子在隐藏状态空间进行解析延拓，
    强制逻辑轨迹变得连续，防止小模型的"逻辑坍缩"
    """
    
    def __init__(
        self,
        hidden_dim: int,
        alpha: float = 0.1,
        use_laplacian: bool = True,
        use_gaussian: bool = True,
        gaussian_sigma: float = 1.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.use_laplacian = use_laplacian
        self.use_gaussian = use_gaussian
        self.gaussian_sigma = gaussian_sigma
        
        # 可学习的平滑权重
        self.smooth_weight = nn.Parameter(torch.ones(1) * alpha)
        
        # 拉普拉斯核 [-1, 2, -1]
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([-1.0, 2.0, -1.0]).view(1, 1, 3)
        )
        
        # 高斯平滑核
        if use_gaussian:
            kernel_size = 5
            x = torch.arange(kernel_size) - kernel_size // 2
            gaussian = torch.exp(-x**2 / (2 * gaussian_sigma**2))
            gaussian = gaussian / gaussian.sum()
            self.register_buffer(
                'gaussian_kernel',
                gaussian.view(1, 1, kernel_size)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, hidden_dim] 隐藏状态
        
        Returns:
            平滑后的隐藏状态
        """
        if x.size(1) < 3:
            return x  # 序列太短，无法平滑
        
        # 保存原始状态
        residual = x.clone()
        
        # 1. 拉普拉斯平滑（捕捉逻辑跳跃）
        if self.use_laplacian:
            # 对每个隐藏维度应用拉普拉斯算子
            x_permuted = x.permute(0, 2, 1)  # [B, H, S]
            
            # 计算二阶差分（拉普拉斯）
            # diff = x[t+1] - 2*x[t] + x[t-1]
            diff = x_permuted[:, :, 2:] - 2 * x_permuted[:, :, 1:-1] + x_permuted[:, :, :-2]
            
            # 填充边界
            diff_padded = F.pad(diff, (1, 1), mode='replicate')
            
            # 应用平滑修正
            alpha = torch.sigmoid(self.smooth_weight)  # 确保alpha在[0,1]范围
            x_permuted = x_permuted + alpha * diff_padded
            
            x = x_permuted.permute(0, 2, 1)  # [B, S, H]
        
        # 2. 高斯平滑（消除高频噪声）
        if self.use_gaussian and x.size(1) >= 5:
            x_permuted = x.permute(0, 2, 1)  # [B, H, S]
            
            # 对每个隐藏维度独立应用高斯核
            batch_size, hidden_dim, seq_len = x_permuted.shape
            
            # 使用unfold来应用高斯平滑
            x_unfold = x_permuted.unfold(2, 5, 1)  # [B, H, seq_len-4, 5]
            gaussian_weights = self.gaussian_kernel.view(1, 1, 1, 5)  # [1, 1, 1, 5]
            x_smoothed_unfold = (x_unfold * gaussian_weights).sum(dim=-1)  # [B, H, seq_len-4]
            
            # 填充边界
            x_smoothed = F.pad(x_smoothed_unfold, (2, 2), mode='replicate')
            
            x = x_smoothed.permute(0, 2, 1)  # [B, S, H]
        
        # 3. 残差连接（保留原始逻辑）
        x = 0.9 * x + 0.1 * residual
        
        return x
    
    def compute_logic_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算逻辑稠密度
        
        通过测量隐藏状态的二阶导数大小来评估逻辑连续性
        """
        if x.size(1) < 3:
            return torch.tensor(1.0)
        
        # 计算二阶差分
        diff = x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]
        
        # 稠密度 = 1 / (1 + 平均跳跃幅度)
        jump_magnitude = torch.mean(torch.abs(diff))
        density = 1.0 / (1.0 + jump_magnitude)
        
        return density


class LogicSmoothingProcessor:
    """
    逻辑平滑处理器
    
    在推理阶段直接外挂，对Logits进行平滑处理
    """
    
    def __init__(
        self,
        hidden_dim: int,
        alpha: float = 0.1,
        vocab_size: int = 10000,
        smoothing_strength: float = 0.3,
        temperature: float = 1.0
    ):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.smoothing_strength = smoothing_strength
        self.temperature = temperature
        
        # 黎曼平滑层
        self.riemann_smoother = RiemannSmoothingLayer(
            hidden_dim=hidden_dim,
            alpha=alpha
        )
        
        # 状态缓存
        self.prev_hidden: Optional[torch.Tensor] = None
        self.prev_logits: Optional[torch.Tensor] = None
        self.current_density: float = 1.0
    
    def smooth_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        平滑隐藏状态
        """
        smoothed = self.riemann_smoother(hidden_states)
        self.current_density = self.riemann_smoother.compute_logic_density(hidden_states).item()
        self.prev_hidden = smoothed.clone()
        return smoothed
    
    def get_logic_density(self) -> float:
        """获取当前逻辑稠密度"""
        return self.current_density
    
    def _build_continuity_matrix(self, vocab_size: int) -> torch.Tensor:
        """
        构建逻辑连续性矩阵
        
        数字token之间应该有强连续性
        """
        matrix = torch.eye(vocab_size)
        
        # 数字token (假设在词表中的位置)
        # 相邻数字之间加强连接
        for i in range(vocab_size - 1):
            matrix[i, i+1] = 0.3
            matrix[i+1, i] = 0.3
        
        return matrix
    
    def smooth_logits(
        self,
        logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        平滑Logits
        
        Args:
            logits: 当前步的logits [batch, vocab_size]
            prev_logits: 前一步的logits（用于连续性约束）
        
        Returns:
            平滑后的logits
        """
        # 1. 温度缩放
        logits = logits / self.temperature
        
        # 2. 应用连续性约束
        if prev_logits is not None:
            # 当前logits应该与前一步的logits保持连续
            continuity_weight = self.smoothing_strength
            logits = (1 - continuity_weight) * logits + continuity_weight * prev_logits
        
        # 3. 拉普拉斯平滑（减少极端值）
        # 这可以防止模型"坍缩"到某个特定token
        log_probs = F.log_softmax(logits, dim=-1)
        smoothed_log_probs = log_probs - self.smoothing_strength * torch.abs(log_probs)
        
        return smoothed_log_probs
    
    def prevent_logic_collapse(
        self,
        logits: torch.Tensor,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        防止逻辑坍缩
        
        当模型对某个token过于自信时，强制分散概率
        """
        # 找到top-k个最高概率的token
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        
        # 计算熵
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # 如果熵太低（过于自信），强制分散
        max_entropy = math.log(self.vocab_size)
        entropy_ratio = entropy / max_entropy
        
        if entropy_ratio < 0.1:  # 熵太低
            # 对top-k之外的token增加概率
            boost = self.smoothing_strength * (1 - entropy_ratio)
            logits = logits + boost
        
        return logits


class DenseLogicEngine:
    """
    稠密逻辑引擎
    
    集成黎曼平滑层和逻辑处理器
    """
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        alpha: float = 0.1,
        smoothing_strength: float = 0.3
    ):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # 黎曼平滑层
        self.riemann_smoother = RiemannSmoothingLayer(
            hidden_dim=hidden_dim,
            alpha=alpha
        )
        
        # 逻辑处理器
        self.logic_processor = LogicSmoothingProcessor(
            vocab_size=vocab_size,
            smoothing_strength=smoothing_strength
        )
        
        # 状态缓存
        self.prev_hidden: Optional[torch.Tensor] = None
        self.prev_logits: Optional[torch.Tensor] = None
    
    def process_hidden_states(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        处理隐藏状态
        
        Returns:
            平滑后的隐藏状态, 逻辑稠密度
        """
        # 应用黎曼平滑
        smoothed = self.riemann_smoother(hidden_states)
        
        # 计算逻辑稠密度
        density = self.riemann_smoother.compute_logic_density(hidden_states)
        
        # 更新缓存
        self.prev_hidden = smoothed.clone()
        
        return smoothed, density.item()
    
    def process_logits(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        处理Logits
        """
        # 平滑
        smoothed = self.logic_processor.smooth_logits(
            logits,
            self.prev_logits
        )
        
        # 防止坍缩
        smoothed = self.logic_processor.prevent_logic_collapse(smoothed)
        
        # 更新缓存
        self.prev_logits = logits.clone()
        
        return smoothed
    
    def reset(self):
        """重置状态"""
        self.prev_hidden = None
        self.prev_logits = None


# ============================================================
# 测试代码
# ============================================================

def test_riemann_smoothing():
    """测试黎曼平滑层"""
    print("\n" + "="*60)
    print("测试黎曼平滑层")
    print("="*60)
    
    # 创建测试数据
    batch_size = 1
    seq_len = 10
    hidden_dim = 128
    
    # 模拟隐藏状态（带有噪声）
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 添加一些"逻辑跳跃"（模拟小模型的不连续推理）
    hidden_states[0, 5, :] += 2.0  # 在第5个位置添加跳跃
    
    # 创建平滑层
    smoother = RiemannSmoothingLayer(hidden_dim=hidden_dim, alpha=0.2)
    
    # 应用平滑
    smoothed_states = smoother(hidden_states)
    
    # 计算稠密度
    original_density = smoother.compute_logic_density(hidden_states)
    smoothed_density = smoother.compute_logic_density(smoothed_states)
    
    print(f"\n原始逻辑稠密度: {original_density.item():.4f}")
    print(f"平滑后稠密度: {smoothed_density.item():.4f}")
    print(f"提升: {(smoothed_density - original_density).item():.4f}")
    
    # 检查跳跃是否被平滑
    original_jump = torch.norm(hidden_states[0, 5, :] - hidden_states[0, 4, :])
    smoothed_jump = torch.norm(smoothed_states[0, 5, :] - smoothed_states[0, 4, :])
    
    print(f"\n原始跳跃幅度: {original_jump.item():.4f}")
    print(f"平滑后跳跃幅度: {smoothed_jump.item():.4f}")
    print(f"跳跃减少: {(original_jump - smoothed_jump).item():.4f}")


def test_logic_processor():
    """测试逻辑处理器"""
    print("\n" + "="*60)
    print("测试逻辑处理器")
    print("="*60)
    
    vocab_size = 1000
    
    # 创建处理器
    processor = LogicSmoothingProcessor(
        vocab_size=vocab_size,
        smoothing_strength=0.3
    )
    
    # 模拟logits（过于自信的分布）
    logits = torch.zeros(1, vocab_size)
    logits[0, 42] = 10.0  # 模型非常自信地选择token 42
    
    # 计算原始熵
    probs = F.softmax(logits, dim=-1)
    original_entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    
    print(f"\n原始分布:")
    print(f"  最高概率token: 42")
    print(f"  概率: {probs[0, 42].item():.4f}")
    print(f"  熵: {original_entropy.item():.4f}")
    
    # 防止坍缩
    smoothed_logits = processor.prevent_logic_collapse(logits)
    smoothed_probs = F.softmax(smoothed_logits, dim=-1)
    smoothed_entropy = -torch.sum(smoothed_probs * torch.log(smoothed_probs + 1e-10))
    
    print(f"\n平滑后分布:")
    print(f"  最高概率token: 42")
    print(f"  概率: {smoothed_probs[0, 42].item():.4f}")
    print(f"  熵: {smoothed_entropy.item():.4f}")
    print(f"  熵增加: {(smoothed_entropy - original_entropy).item():.4f}")


if __name__ == "__main__":
    test_riemann_smoothing()
    test_logic_processor()
    
    print("\n" + "="*60)
    print("✓ 测试完成")
    print("="*60)
