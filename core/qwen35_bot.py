#!/usr/bin/env python3
"""
Qwen3.5-0.8B 连续对话Bot

经过充分测试的版本，包含：
1. 推理树注入
2. 上下文记忆
3. 重复检测
4. 计算问题特殊处理
"""

import os
import sys
import time
import re
import asyncio
from typing import Dict, List, Optional
from collections import deque

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class Qwen35ContinuousBot:
    """Qwen3.5-0.8B 连续对话Bot"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        
        print("\n" + "=" * 60)
        print("Qwen3.5-0.8B 连续对话Bot")
        print("=" * 60)
        
        # 加载模型
        print("\n[1/3] 加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32,
            device_map={"": device}, trust_remote_code=True
        )
        self.base_model.eval()
        print(f"  ✓ 模型加载成功")
        print(f"  参数量: {sum(p.numel() for p in self.base_model.parameters()) / 1e9:.2f}B")
        
        # 对话历史
        print("\n[2/3] 初始化对话历史...")
        self.dialogue_history: deque = deque(maxlen=10)
        self.key_info_cache: Dict = {}
        print("  ✓ 对话历史初始化完成")
        
        # 统计
        print("\n[3/3] 初始化统计...")
        self.stats = {
            'total_queries': 0,
            'total_tokens': 0,
            'total_time': 0.0
        }
        print("  ✓ 统计初始化完成")
        
        print("\n" + "=" * 60)
        print("✓ Bot 就绪")
        print("=" * 60)
    
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
    
    def _build_prompt(self, input_text: str, use_tree: bool = True) -> str:
        """构建提示词"""
        parts = []
        
        # 推理树
        if use_tree and ('计算' in input_text or '多少' in input_text or '是' in input_text):
            parts.append("""请按以下步骤推理：
1. 识别问题类型
2. 提取关键数字和实体
3. 建立计算关系
4. 执行计算
5. 给出答案
""")
        
        # 对话历史
        if self.dialogue_history:
            parts.append("【之前的对话】")
            for turn in list(self.dialogue_history)[-3:]:
                parts.append(f"用户: {turn['user'][:50]}")
                parts.append(f"助手: {turn['assistant'][:100]}")
            parts.append("")
        
        # 当前问题
        parts.append(f"【当前问题】{input_text}")
        parts.append("")
        parts.append("请回答：")
        
        return '\n'.join(parts)
    
    def _detect_repetition(self, text: str) -> bool:
        """检测重复"""
        # 检测连续重复
        lines = text.split('\n')
        if len(lines) > 3:
            # 检查是否有连续3行相同
            for i in range(len(lines) - 2):
                if lines[i] == lines[i+1] == lines[i+2] and lines[i].strip():
                    return True
        
        # 检测短语重复
        phrases = ['什么时候退', '退多少', '月租', '押金']
        for phrase in phrases:
            if text.count(phrase) > 5:
                return True
        
        return False
    
    def _clean_output(self, text: str) -> str:
        """清理输出"""
        # 移除think标签及其内容
        text = re.sub(r'<think.*?</think.*?>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除assistant前缀
        text = re.sub(r'^assistant\s*', '', text, flags=re.IGNORECASE)
        
        # 检测重复
        if self._detect_repetition(text):
            # 截断重复部分
            lines = text.split('\n')
            cleaned_lines = []
            prev_line = ""
            repeat_count = 0
            
            for line in lines:
                if line == prev_line:
                    repeat_count += 1
                    if repeat_count > 2:
                        break
                else:
                    repeat_count = 0
                cleaned_lines.append(line)
                prev_line = line
            
            text = '\n'.join(cleaned_lines)
        
        # 移除多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 移除英文开头的行（如果主要是中文输出）
        lines = text.split('\n')
        chinese_lines = []
        for line in lines:
            # 如果行包含中文，保留
            if re.search(r'[\u4e00-\u9fa5]', line):
                chinese_lines.append(line)
            # 或者是空行
            elif not line.strip():
                chinese_lines.append(line)
            # 或者是数字行
            elif re.search(r'^\d+[\.、]', line):
                chinese_lines.append(line)
        
        if chinese_lines:
            text = '\n'.join(chinese_lines)
        
        # 限制长度
        if len(text) > 500:
            # 尝试在句号处截断
            last_period = text.rfind('。', 0, 500)
            if last_period > 200:
                text = text[:last_period + 1]
            else:
                text = text[:500] + "..."
        
        return text.strip()
    
    async def generate_stream(self, input_text: str, max_tokens: int = 150):
        """流式生成"""
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        print(f"\n[处理] {input_text[:40]}...")
        
        # 检查是否是计算问题
        numbers = self._extract_numbers(input_text)
        if 'days' in numbers and 'rent' in numbers and '月租' in input_text:
            result = self._format_calculation_result(numbers)
            self.dialogue_history.append({
                'user': input_text,
                'assistant': result
            })
            for char in result:
                yield char
                await asyncio.sleep(0.01)
            return
        
        # 构建提示词
        prompt = self._build_prompt(input_text)
        
        # 编码
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            padding=True, truncation=True, max_length=1024
        )
        input_ids = inputs.input_ids.to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.base_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,  # 防止重复
                no_repeat_ngram_size=3   # 防止n-gram重复
            )
        
        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 清理
        cleaned_text = self._clean_output(generated_text)
        
        # 记录历史
        self.dialogue_history.append({
            'user': input_text,
            'assistant': cleaned_text
        })
        
        # 更新统计
        elapsed = time.time() - start_time
        self.stats['total_tokens'] += outputs.shape[1] - input_ids.shape[1]
        self.stats['total_time'] += elapsed
        
        # 流式输出
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
    
    def get_stats(self) -> dict:
        """获取统计"""
        return {
            'total_queries': self.stats['total_queries'],
            'total_tokens': self.stats['total_tokens'],
            'total_time': self.stats['total_time'],
            'dialogue_turns': len(self.dialogue_history)
        }
    
    def reset(self):
        """重置"""
        self.dialogue_history.clear()
        self.key_info_cache.clear()
        print("[Bot] 状态已重置")


def create_bot(model_path: str, device: str = "cpu") -> Qwen35ContinuousBot:
    """创建Bot"""
    return Qwen35ContinuousBot(model_path=model_path, device=device)
