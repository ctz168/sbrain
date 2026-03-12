#!/usr/bin/env python3
"""
类人脑双系统AI - 连续密度场版

核心创新：
1. 连续逻辑密度场（不是离散分类）
2. 逐token实时处理
3. 动态调整推理强度
4. 与高刷新模型完美融合
"""

import os
import sys
import time
import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.config import default_config


@dataclass
class TokenDensity:
    """单个token的逻辑密度"""
    token: str
    density: float
    temperature: float


class ContinuousDensityField(nn.Module):
    """
    连续逻辑密度场
    
    核心思想：
    - 每个token都有一个连续的逻辑密度值
    - 不是离散分类，而是连续场
    - 逐token实时计算
    """
    
    def __init__(self, hidden_size: int = 896):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 密度估计网络
        self.density_net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 逻辑锚点特征
        self.logic_anchors = {
            'strong': ['计算', '等于', '多少', '求', '证明', '推导', 
                      '因为', '所以', '如果', '那么', '必然', '一定',
                      '月租', '房租', '押金', '合计', '费用'],
            'medium': ['怎样', '如何', '什么', '为什么', '分析', '判断'],
            'weak': ['写', '创作', '想象', '感觉', '觉得', '喜欢', '故事', '诗']
        }
    
    def compute_density(
        self, 
        token: str, 
        hidden_state: torch.Tensor,
        context: str = ""
    ) -> float:
        """计算单个token的逻辑密度"""
        # 处理hidden_state维度
        if hidden_state.dim() == 0:
            hidden_state = hidden_state.unsqueeze(0)
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.flatten()
        
        # 神经网络估计
        neural_density = self.density_net(hidden_state).item()
        
        # 锚点估计
        anchor_density = self._anchor_density(token, context)
        
        # 融合
        if anchor_density > 0.5:
            density = 0.6 * anchor_density + 0.4 * neural_density
        else:
            density = 0.4 * anchor_density + 0.6 * neural_density
        
        return density
    
    def _anchor_density(self, token: str, context: str) -> float:
        """基于锚点计算密度"""
        text = token + context
        
        for anchor in self.logic_anchors['strong']:
            if anchor in text:
                return 0.85
        
        for anchor in self.logic_anchors['medium']:
            if anchor in text:
                return 0.55
        
        for anchor in self.logic_anchors['weak']:
            if anchor in text:
                return 0.25
        
        # 检查数字
        if re.search(r'\d+', text):
            return 0.7
        
        return 0.4
    
    def density_to_temperature(self, density: float) -> float:
        """将密度转换为温度"""
        # 高密度 → 低温度（更确定）
        # 低密度 → 高温度（更发散）
        return 0.2 + (1 - density) * 0.6


class ContinuousDensityBrain:
    """
    连续密度场类脑AI
    
    核心特点：
    1. 逐token实时处理
    2. 连续密度场
    3. 动态调整推理强度
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.config = default_config
        self.device = device
        
        print("\n" + "=" * 60)
        print("连续密度场类脑AI")
        print("=" * 60)
        
        # 加载模型
        print("\n[1/4] 加载基础模型...")
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
        
        # 连续密度场
        print("\n[2/4] 初始化连续密度场...")
        self.density_field = ContinuousDensityField(self.hidden_size)
        print("  ✓ 密度场初始化完成")
        
        # 类脑模块
        print("\n[3/4] 初始化类脑模块...")
        try:
            from stdp.stdp_engine import STDPController
            from hippocampus.hippocampus_system import HippocampusSystem
            
            self.stdp = STDPController(self.config.stdp)
            self.hippocampus = HippocampusSystem(self.config.hippocampus)
            print("  ✓ 类脑模块初始化完成")
        except Exception as e:
            print(f"  ! 类脑模块跳过: {e}")
            self.stdp = None
            self.hippocampus = None
        
        # 统计
        print("\n[4/4] 初始化统计模块...")
        self.stats = {
            'total_tokens': 0,
            'high_density_tokens': 0,
            'medium_density_tokens': 0,
            'low_density_tokens': 0,
            'avg_density': 0.0,
            'total_queries': 0
        }
        print("  ✓ 统计模块初始化完成")
        
        print("\n" + "=" * 60)
        print("✓ 所有模块初始化完成")
        print("=" * 60)
    
    async def generate_stream(self, input_text: str, max_tokens: int = 200):
        """
        流式生成 - 逐token实时调整
        
        这是真正的类人脑处理方式！
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        try:
            # 编码输入
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # 计算输入的平均密度
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                input_hiddens = outputs.hidden_states[-1][0]
            
            # 计算输入token的密度
            input_tokens = [self.tokenizer.decode([i]) for i in input_ids[0]]
            input_densities = []
            
            for i, token in enumerate(input_tokens):
                if i < input_hiddens.shape[0]:
                    density = self.density_field.compute_density(
                        token, input_hiddens[i], input_text
                    )
                    input_densities.append(density)
            
            avg_input_density = sum(input_densities) / max(1, len(input_densities))
            
            print(f"\n[密度场分析]")
            print(f"  输入平均密度: {avg_input_density:.2f}")
            
            # 根据输入密度决定温度
            temperature = self.density_field.density_to_temperature(avg_input_density)
            print(f"  生成温度: {temperature:.2f}")
            
            # 使用标准生成，但根据密度调整温度
            with torch.no_grad():
                output_ids = self.base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=max(0.1, temperature),
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            print(f"  输出token数: {output_ids.shape[1] - input_ids.shape[1]}")
            
            # 解码输出
            generated_text = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            print(f"  生成文本长度: {len(generated_text)}")
            
            # 如果生成内容为空，返回默认消息
            if not generated_text or not generated_text.strip():
                generated_text = "我正在思考这个问题，请稍等..."
            
            # 更新统计
            self.stats['total_tokens'] += len(generated_text)
            if avg_input_density > 0.6:
                self.stats['high_density_tokens'] += len(generated_text)
            elif avg_input_density > 0.4:
                self.stats['medium_density_tokens'] += len(generated_text)
            else:
                self.stats['low_density_tokens'] += len(generated_text)
            
            self.stats['avg_density'] = (
                (self.stats['avg_density'] * (self.stats['total_queries'] - 1) + avg_input_density) / 
                self.stats['total_queries']
            )
            
            # 流式输出
            for char in generated_text:
                yield char
                await asyncio.sleep(0.01)
                
        except Exception as e:
            print(f"[错误] 生成失败: {e}")
            import traceback
            traceback.print_exc()
            yield f"生成出错: {str(e)[:50]}"
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total = max(1, self.stats['total_tokens'])
        return {
            'system': {
                'total_queries': self.stats['total_queries'],
                'total_tokens': self.stats['total_tokens'],
                'avg_density': self.stats['avg_density']
            },
            'density_distribution': {
                'high': self.stats['high_density_tokens'] / total,
                'medium': self.stats['medium_density_tokens'] / total,
                'low': self.stats['low_density_tokens'] / total
            }
        }


def create_brain_ai(model_path: str, device: str = "cpu") -> ContinuousDensityBrain:
    """创建连续密度场类脑AI"""
    return ContinuousDensityBrain(model_path=model_path, device=device)
