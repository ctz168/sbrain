#!/usr/bin/env python3
"""
模型加载与窄宽带控制

核心功能：
1. 加载Qwen3.5-0.8B模型
2. 实现窄宽带过滤器
3. 控制生成参数（Greedy Search）
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class NarrowBandwidthConfig:
    """窄宽带配置"""
    max_new_tokens: int = 24        # 极短生成（高刷新）
    window_size: int = 2            # 窗口大小：只保留前1步的结论
    temperature: float = 1.0        # Greedy Search (temp=1.0)
    do_sample: bool = False         # 关闭采样，使用Greedy
    top_p: float = 1.0              # 不使用nucleus sampling
    repetition_penalty: float = 1.0 # 不使用重复惩罚


class ModelHandler:
    """
    模型处理器
    
    核心特性：
    1. 窄宽带过滤：每次只喂入上一个微步的结论
    2. Greedy Search：保证逻辑确定性
    3. 高刷新生成：max_new_tokens=16-32
    """
    
    def __init__(
        self,
        model_path: str = "../models/Qwen3.5-0.8B",
        device: str = "cpu",
        config: Optional[NarrowBandwidthConfig] = None
    ):
        self.device = device
        self.config = config or NarrowBandwidthConfig()
        
        print(f"\n{'='*60}")
        print("LSDC 引擎 - 模型加载器")
        print("="*60)
        
        # 加载模型
        print(f"\n[1/2] 加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": device},
            trust_remote_code=True
        )
        self.model.eval()
        
        self.hidden_size = self.model.config.hidden_size
        print(f"  ✓ 模型加载成功")
        print(f"  参数量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        print(f"  隐藏层: {self.hidden_size}")
        
        # 窄宽带状态
        print(f"\n[2/2] 初始化窄宽带控制器")
        print(f"  max_new_tokens: {self.config.max_new_tokens}")
        print(f"  window_size: {self.config.window_size}")
        print(f"  do_sample: {self.config.do_sample} (Greedy)")
        print(f"  ✓ 窄宽带控制器初始化完成")
        
        print(f"\n{'='*60}")
        print("✓ 模型加载完成")
        print("="*60)
    
    def narrow_bandwidth_filter(
        self,
        current_goal: str,
        previous_conclusion: Optional[str] = None
    ) -> str:
        """
        窄宽带过滤器
        
        核心逻辑：
        - 只保留上一个微步的 Conclusion
        - 和当前的 Goal
        - 丢弃所有历史过程
        
        Args:
            current_goal: 当前目标
            previous_conclusion: 上一步的结论
        
        Returns:
            过滤后的提示词
        """
        if previous_conclusion:
            # 窄宽带：只保留前一步结论 + 当前目标
            prompt = f"""已知: {previous_conclusion}

问题: {current_goal}

请直接回答，不要重复问题："""
        else:
            # 第一步：只有目标
            prompt = f"""问题: {current_goal}

请直接回答，不要重复问题："""
        
        return prompt
    
    def generate_micro_step(
        self,
        prompt: str
    ) -> Tuple[str, torch.Tensor]:
        """
        生成微步
        
        使用Greedy Search保证逻辑确定性
        
        Args:
            prompt: 输入提示词
        
        Returns:
            generated_text: 生成的文本
            hidden_state: 隐藏状态（用于状态连续性）
        """
        # 编码
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # 生成（Greedy Search）
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 获取隐藏状态（用于状态连续性）
        with torch.no_grad():
            hidden_outputs = self.model(
                input_ids=outputs,
                attention_mask=torch.ones_like(outputs),
                output_hidden_states=True,
                return_dict=True
            )
            hidden_state = hidden_outputs.hidden_states[-1][:, -1, :]
        
        return generated_text, hidden_state
    
    def extract_conclusion(self, text: str) -> str:
        """
        从生成文本中提取结论
        
        用于下一步的窄宽带过滤
        """
        # 简单实现：取最后一句话
        sentences = text.replace('。', '。\n').split('\n')
        for s in reversed(sentences):
            s = s.strip()
            if s and len(s) > 2:
                return s
        return text.strip()[-50:] if len(text) > 50 else text.strip()
    
    def is_logic_dense(self, text: str) -> bool:
        """
        检查逻辑稠密性
        
        如果跨度过大，返回False，需要补齐
        """
        # 简单实现：检查是否有推理连接词
        reasoning_words = ['因为', '所以', '因此', '由于', '导致', '使得', '从而']
        has_reasoning = any(w in text for w in reasoning_words)
        
        # 检查长度（太短可能跳过了步骤）
        is_long_enough = len(text) > 10
        
        return has_reasoning and is_long_enough


def create_model_handler(
    model_path: str = "../models/Qwen3.5-0.8B",
    device: str = "cpu"
) -> ModelHandler:
    """创建模型处理器"""
    return ModelHandler(model_path=model_path, device=device)
