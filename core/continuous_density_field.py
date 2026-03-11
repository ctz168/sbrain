#!/usr/bin/env python3
"""
连续逻辑密度场系统

核心思想：
1. 不是离散分类，而是连续密度值
2. 每个token都有一个"逻辑密度"
3. 高刷新模型逐token实时处理
4. 动态调整推理强度

这更符合人脑的工作方式！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class TokenDensity:
    """单个token的逻辑密度"""
    token: str
    density: float  # 0.0 ~ 1.0
    reason: str     # 密度来源


class ContinuousLogicDensityField(nn.Module):
    """
    连续逻辑密度场
    
    核心创新：
    1. 每个token都有一个连续的逻辑密度值
    2. 不是离散分类，而是连续场
    3. 逐token实时计算
    4. 与高刷新模型完美融合
    """
    
    def __init__(self, hidden_size: int = 896):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 密度估计网络：从隐状态估计逻辑密度
        self.density_estimator = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出 0~1 的密度值
        )
        
        # 密度平滑：相邻token的密度应该平滑过渡
        self.smoothing_kernel = nn.Parameter(
            torch.ones(5) / 5,  # 5点平滑
            requires_grad=False
        )
        
        # 逻辑锚点特征库
        self.logic_anchors = {
            # 强逻辑锚点 (密度 > 0.8)
            'strong': ['计算', '等于', '多少', '求', '证明', '推导', 
                      '因为', '所以', '如果', '那么', '必然'],
            # 中等逻辑锚点 (密度 0.5-0.8)
            'medium': ['分析', '判断', '比较', '原因', '结果', 
                      '规则', '条件', '方法', '步骤'],
            # 弱逻辑锚点 (密度 0.2-0.5)
            'weak': ['怎样', '如何', '什么', '为什么', '可能'],
            # 无逻辑锚点 (密度 < 0.2)
            'none': ['写', '创作', '想象', '感觉', '觉得', '喜欢']
        }
    
    def compute_token_density(
        self, 
        token: str, 
        hidden_state: torch.Tensor,
        context: str = ""
    ) -> float:
        """
        计算单个token的逻辑密度
        
        这是实时的、连续的！
        
        Args:
            token: 当前token
            hidden_state: 当前隐状态
            context: 上下文
        
        Returns:
            density: 逻辑密度值 [0, 1]
        """
        # 方法1：基于隐状态的估计
        if hidden_state.dim() == 0:
            hidden_state = hidden_state.unsqueeze(0)
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.flatten()
        neural_density = self.density_estimator(hidden_state).item()
        
        # 方法2：基于锚点的估计
        anchor_density = self._anchor_based_density(token, context)
        
        # 融合两种估计
        # 如果有明确的锚点，更信任锚点
        if anchor_density > 0.5:
            density = 0.7 * anchor_density + 0.3 * neural_density
        else:
            density = 0.5 * anchor_density + 0.5 * neural_density
        
        return density
    
    def compute_sequence_density(
        self,
        tokens: List[str],
        hidden_states: torch.Tensor
    ) -> List[TokenDensity]:
        """
        计算整个序列的逻辑密度场
        
        Args:
            tokens: token列表
            hidden_states: 隐状态 [seq_len, hidden_size]
        
        Returns:
            densities: 每个token的密度
        """
        raw_densities = []
        
        for i, token in enumerate(tokens):
            density = self.compute_token_density(
                token, 
                hidden_states[i],
                context=''.join(tokens[max(0,i-5):i+5])
            )
            raw_densities.append(density)
        
        # 平滑处理
        smoothed = self._smooth_densities(raw_densities)
        
        # 构建结果
        results = []
        for i, token in enumerate(tokens):
            results.append(TokenDensity(
                token=token,
                density=smoothed[i],
                reason=self._explain_density(token, smoothed[i])
            ))
        
        return results
    
    def _anchor_based_density(self, token: str, context: str) -> float:
        """基于锚点计算密度"""
        # 检查强锚点
        for anchor in self.logic_anchors['strong']:
            if anchor in token or anchor in context:
                return 0.85
        
        # 检查中等锚点
        for anchor in self.logic_anchors['medium']:
            if anchor in token or anchor in context:
                return 0.65
        
        # 检查弱锚点
        for anchor in self.logic_anchors['weak']:
            if anchor in token or anchor in context:
                return 0.35
        
        # 检查无锚点
        for anchor in self.logic_anchors['none']:
            if anchor in token or anchor in context:
                return 0.15
        
        # 默认中等密度
        return 0.4
    
    def _smooth_densities(self, densities: List[float]) -> List[float]:
        """平滑密度序列"""
        if len(densities) < 5:
            return densities
        
        smoothed = []
        for i in range(len(densities)):
            # 取周围5个点的加权平均
            start = max(0, i - 2)
            end = min(len(densities), i + 3)
            window = densities[start:end]
            smoothed.append(sum(window) / len(window))
        
        return smoothed
    
    def _explain_density(self, token: str, density: float) -> str:
        """解释密度来源"""
        if density > 0.7:
            return f"强逻辑需求: '{token}'"
        elif density > 0.5:
            return f"中等逻辑需求: '{token}'"
        elif density > 0.3:
            return f"弱逻辑需求: '{token}'"
        else:
            return f"无逻辑需求: '{token}'"


class DynamicReasoningAdjuster(nn.Module):
    """
    动态推理调整器
    
    根据逻辑密度实时调整推理强度
    """
    
    def __init__(self, hidden_size: int = 896):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 推理强度控制器
        self.intensity_controller = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),  # +1 for density
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 温度调整器
        self.temperature_adjuster = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def adjust(
        self,
        hidden_state: torch.Tensor,
        density: float
    ) -> Tuple[torch.Tensor, float]:
        """
        根据密度调整推理
        
        Args:
            hidden_state: 当前隐状态
            density: 逻辑密度
        
        Returns:
            adjusted_hidden: 调整后的隐状态
            temperature: 调整后的温度
        """
        # 构建密度特征
        density_tensor = torch.tensor([density]).to(hidden_state.device)
        
        # 调整隐状态
        combined = torch.cat([
            hidden_state,
            density_tensor.expand(hidden_state.shape[0], -1)
        ], dim=-1)
        
        adjustment = self.intensity_controller(combined)
        adjusted_hidden = hidden_state + density * adjustment
        
        # 调整温度
        # 高密度 → 低温度（更确定）
        # 低密度 → 高温度（更发散）
        base_temp = 0.7
        temp_adjustment = self.temperature_adjuster(density_tensor.unsqueeze(0))
        temperature = base_temp + (1 - density) * 0.5 * temp_adjustment.item()
        
        return adjusted_hidden, temperature


class ContinuousFieldBrain:
    """
    连续逻辑密度场类脑系统
    
    核心特点：
    1. 逐token实时处理
    2. 连续密度场，不是离散分类
    3. 动态调整推理强度
    4. 与高刷新模型完美融合
    """
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 获取隐藏层大小
        self.hidden_size = model.config.hidden_size
        
        # 连续密度场
        self.density_field = ContinuousLogicDensityField(self.hidden_size)
        
        # 动态调整器
        self.adjuster = DynamicReasoningAdjuster(self.hidden_size)
        
        # 统计
        self.stats = {
            'total_tokens': 0,
            'high_density_tokens': 0,
            'low_density_tokens': 0
        }
    
    async def generate_stream(self, input_text: str, max_tokens: int = 200):
        """
        流式生成，逐token实时调整
        
        这是真正的类人脑处理方式！
        """
        # 编码输入
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        # 获取输入的密度场
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True
            )
            input_hiddens = outputs.hidden_states[-1][0]  # [seq_len, hidden]
        
        input_tokens = [self.tokenizer.decode([i]) for i in input_ids[0]]
        input_densities = self.density_field.compute_sequence_density(
            input_tokens, input_hiddens
        )
        
        # 计算输入的平均密度
        avg_input_density = sum(d.density for d in input_densities) / len(input_densities)
        
        print(f"\n[密度场分析]")
        print(f"  输入平均密度: {avg_input_density:.2f}")
        for i, d in enumerate(input_densities[-5:]):
            print(f"  Token '{d.token}': 密度={d.density:.2f}")
        
        # 生成
        generated_ids = input_ids.clone()
        current_density = avg_input_density
        
        import asyncio
        import torch
        
        for _ in range(max_tokens):
            with torch.no_grad():
                # 获取当前隐状态
                outputs = self.model(
                    input_ids=generated_ids,
                    output_hidden_states=True
                )
                last_hidden = outputs.hidden_states[-1][0, -1, :]  # [hidden]
                logits = outputs.logits[0, -1, :]  # [vocab]
                
                # 计算当前token的密度
                current_token = self.tokenizer.decode([logits.argmax()])
                current_density = self.density_field.compute_token_density(
                    current_token, last_hidden.unsqueeze(0)
                )
                
                # 根据密度调整温度
                if current_density > 0.7:
                    temperature = 0.2  # 高密度，低温度，更确定
                elif current_density > 0.4:
                    temperature = 0.5  # 中密度
                else:
                    temperature = 0.8  # 低密度，高温度，更发散
                
                # 采样下一个token
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # 更新统计
                self.stats['total_tokens'] += 1
                if current_density > 0.6:
                    self.stats['high_density_tokens'] += 1
                else:
                    self.stats['low_density_tokens'] += 1
                
                # 检查结束
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # 追加token
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # 流式输出
                decoded = self.tokenizer.decode(next_token)
                yield decoded
                await asyncio.sleep(0.01)
    
    def get_density_field_stats(self) -> dict:
        """获取密度场统计"""
        return {
            'total_tokens': self.stats['total_tokens'],
            'high_density_ratio': (
                self.stats['high_density_tokens'] / max(1, self.stats['total_tokens'])
            ),
            'low_density_ratio': (
                self.stats['low_density_tokens'] / max(1, self.stats['total_tokens'])
            )
        }


# 测试
if __name__ == "__main__":
    print("=" * 60)
    print("连续逻辑密度场系统测试")
    print("=" * 60)
    
    # 创建密度场
    density_field = ContinuousLogicDensityField(hidden_size=896)
    
    # 测试不同类型的token
    test_cases = [
        # 高密度
        ("计算", "月租是多少"),
        ("等于", "1600除以20等于多少"),
        ("因为", "因为A大于B"),
        
        # 中密度
        ("怎样", "押金怎样才能退"),
        ("如何", "如何计算月租"),
        
        # 低密度
        ("写", "写一首诗"),
        ("想象", "想象一下未来"),
        ("感觉", "我感觉今天天气不错"),
    ]
    
    print("\n单Token密度测试:")
    print("-" * 40)
    
    for token, context in test_cases:
        # 模拟隐状态
        hidden = torch.randn(1, 896)
        density = density_field.compute_token_density(token, hidden.squeeze(0), context)
        
        print(f"Token: '{token}'")
        print(f"  Context: {context}")
        print(f"  Density: {density:.2f}")
        print()
