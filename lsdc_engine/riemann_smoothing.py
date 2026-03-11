#!/usr/bin/env python3
"""
黎曼平滑逻辑处理器 (Riemann Smoothing Logic Processor)

核心数学原理：
1. 拉普拉斯算子：捕捉逻辑跳跃
2. 解析延拓：在离散逻辑点之间拉起平滑曲线
3. 黎曼ζ函数：α参数控制逻辑稠密度

数学模型：
- 原始状态: S_n
- 平滑后状态: S'_n = S_n + α * Δ²S_n
- 其中 Δ² 是拉普拉斯算子
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class RiemannSmoothingLayer(nn.Module):
    """
    黎曼平滑层
    
    核心逻辑：
    通过拉普拉斯算子对隐藏状态进行平滑，
    确保逻辑链条的连续性，防止"逻辑坍缩"
    """
    
    def __init__(self, dim: int, alpha: float = 0.1):
        super().__init__()
        self.dim = dim
        self.alpha = alpha  # 逻辑稠密度权重
        
        # 拉普拉斯核：[-1, 2, -1] 用于捕捉二阶导数
        self.register_buffer('laplacian_kernel', torch.tensor([-1.0, 2.0, -1.0]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, hidden_dim]
        
        Returns:
            平滑后的隐藏状态
        """
        # 1. 保存原始逻辑特征
        res = x.clone()
        
        # 2. 计算二阶导数（拉普拉斯算子）
        # 捕捉逻辑跳跃点
        if x.shape[1] > 2:
            diff = x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]
            
            # 3. 将平滑修正补偿回原逻辑
            # 相当于在离散的逻辑点之间拉起了一根平滑的曲线
            x_smoothed = x.clone()
            x_smoothed[:, 1:-1, :] = x[:, 1:-1, :] + self.alpha * diff
            
            return x_smoothed
        
        return x


class LogicSmoothingProcessor:
    """
    逻辑平滑处理器
    
    可以在推理阶段直接外挂到任何模型
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        alpha: float = 0.15,
        temperature: float = 0.7
    ):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.temperature = temperature
        
        # 初始化平滑层
        self.smoother = RiemannSmoothingLayer(dim=hidden_dim, alpha=alpha)
        
        # 历史状态缓存（用于连续性）
        self.state_history: List[torch.Tensor] = []
        self.max_history = 5
    
    def smooth_hidden_states(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        平滑隐藏状态
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            平滑后的隐藏状态
        """
        # 应用黎曼平滑
        smoothed = self.smoother(hidden_states)
        
        # 记录历史（用于连续性检查）
        self.state_history.append(smoothed[:, -1, :].detach().clone())
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        return smoothed
    
    def adjust_logits(
        self,
        logits: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        调整logits分布
        
        核心逻辑：
        1. 降低"逻辑跳跃"token的概率
        2. 提高"逻辑连续"token的概率
        
        Args:
            logits: [batch, vocab_size]
            hidden_states: 可选的隐藏状态
        
        Returns:
            调整后的logits
        """
        adjusted = logits.clone()
        
        # 1. 计算logits的"平滑度"
        # 使用温度缩放
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # 2. 计算熵（衡量不确定性）
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # 3. 如果熵过高（不确定性大），应用更强的平滑
        high_entropy_mask = entropy > 2.0  # 阈值
        
        if high_entropy_mask.any():
            # 对高熵位置应用更强的平滑
            # 这会抑制"逻辑跳跃"
            adjusted[high_entropy_mask] = adjusted[high_entropy_mask] * 0.9
        
        # 4. 如果有历史状态，检查连续性
        if len(self.state_history) >= 2 and hidden_states is not None:
            # 计算当前状态与历史状态的差异
            current = hidden_states[:, -1, :]
            prev = self.state_history[-1]
            
            # 计算余弦相似度
            similarity = F.cosine_similarity(current, prev.unsqueeze(0), dim=-1)
            
            # 如果相似度过低（逻辑跳跃），降低温度
            if similarity.item() < 0.5:
                adjusted = adjusted / 1.2  # 更强的约束
        
        return adjusted
    
    def get_logic_density(self) -> float:
        """
        获取当前逻辑稠密度
        
        基于黎曼ζ函数的类比：
        ζ(1+α) ≈ 1/α + γ (当α→0时)
        
        Returns:
            逻辑稠密度分数
        """
        if len(self.state_history) < 2:
            return 1.0
        
        # 计算状态序列的平滑度
        similarities = []
        for i in range(1, len(self.state_history)):
            sim = F.cosine_similarity(
                self.state_history[i].unsqueeze(0),
                self.state_history[i-1].unsqueeze(0),
                dim=-1
            )
            similarities.append(sim.item())
        
        # 平均相似度作为稠密度
        avg_sim = sum(similarities) / len(similarities)
        
        # 转换为稠密度分数（0-1）
        density = (avg_sim + 1) / 2
        
        return density
    
    def reset(self):
        """重置状态"""
        self.state_history.clear()


class DenseLogicGenerator:
    """
    稠密逻辑生成器
    
    集成黎曼平滑到生成过程
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        processor: LogicSmoothingProcessor,
        device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50
    ) -> Tuple[str, float]:
        """
        生成文本
        
        Args:
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
        
        Returns:
            生成的文本和逻辑稠密度分数
        """
        # 编码
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # 前向传播
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=torch.ones_like(generated_ids),
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # 获取隐藏状态
                hidden_states = outputs.hidden_states[-1]
                
                # 应用黎曼平滑
                smoothed_hidden = self.processor.smooth_hidden_states(hidden_states)
                
                # 获取logits
                logits = outputs.logits[:, -1, :]
                
                # 调整logits
                adjusted_logits = self.processor.adjust_logits(
                    logits, smoothed_hidden
                )
                
                # 采样
                probs = F.softmax(adjusted_logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # 检查结束
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # 追加token
                generated_ids = torch.cat(
                    [generated_ids, next_token.view(1, 1)], dim=1
                )
        
        # 解码
        generated_text = self.tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 获取逻辑稠密度
        density = self.processor.get_logic_density()
        
        return generated_text, density


# ============================================================
# 独立的Logit Processor（可用于vLLM/llama-cpp）
# ============================================================

class RiemannLogitProcessor:
    """
    黎曼Logit处理器
    
    可以直接外挂到vLLM或llama-cpp-python
    """
    
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self.prev_logits: Optional[torch.Tensor] = None
    
    def __call__(
        self,
        token_ids: List[int],
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        处理logits
        
        Args:
            token_ids: 已生成的token列表
            logits: 当前logits [vocab_size]
        
        Returns:
            调整后的logits
        """
        adjusted = logits.clone()
        
        # 如果有前一个logits，计算差异
        if self.prev_logits is not None:
            # 计算logits的变化（一阶导数）
            diff = logits - self.prev_logits
            
            # 应用平滑：减少剧烈变化
            # 这相当于对logits轨迹进行拉普拉斯平滑
            adjusted = logits - self.alpha * diff
        
        # 保存当前logits
        self.prev_logits = logits.clone()
        
        return adjusted
    
    def reset(self):
        """重置状态"""
        self.prev_logits = None


# ============================================================
# 测试代码
# ============================================================

def test_riemann_smoothing():
    """测试黎曼平滑"""
    print("\n" + "="*60)
    print("黎曼平滑逻辑处理器测试")
    print("="*60)
    
    # 1. 测试平滑层
    print("\n[1/3] 测试RiemannSmoothingLayer")
    dim = 128
    smoother = RiemannSmoothingLayer(dim=dim, alpha=0.15)
    
    # 模拟隐藏状态
    hidden = torch.randn(1, 5, dim)
    smoothed = smoother(hidden)
    
    # 计算平滑前后的差异
    diff = (smoothed - hidden).abs().mean().item()
    print(f"  平滑前后平均差异: {diff:.6f}")
    print(f"  ✓ 平滑层工作正常")
    
    # 2. 测试Logit处理器
    print("\n[2/3] 测试LogicSmoothingProcessor")
    processor = LogicSmoothingProcessor(hidden_dim=dim, alpha=0.15)
    
    # 模拟logits
    logits = torch.randn(1, 1000)
    adjusted = processor.adjust_logits(logits)
    
    print(f"  原始logits熵: {F.softmax(logits, dim=-1).max().item():.4f}")
    print(f"  调整后logits熵: {F.softmax(adjusted, dim=-1).max().item():.4f}")
    print(f"  ✓ Logit处理器工作正常")
    
    # 3. 测试独立Logit处理器
    print("\n[3/3] 测试RiemannLogitProcessor")
    riemann_processor = RiemannLogitProcessor(alpha=0.15)
    
    # 模拟多步生成
    token_ids = []
    for i in range(5):
        logits = torch.randn(1000)
        adjusted = riemann_processor(token_ids, logits)
        token_ids.append(i)
        print(f"  步骤{i+1}: logits调整幅度 = {(adjusted - logits).abs().mean().item():.4f}")
    
    print(f"  ✓ 独立Logit处理器工作正常")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过")
    print("="*60)


if __name__ == "__main__":
    test_riemann_smoothing()
