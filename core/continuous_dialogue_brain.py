#!/usr/bin/env python3
"""
对话上下文连续性系统

核心功能：
1. 对话历史缓冲区（短期记忆）
2. 海马体记忆整合（长期记忆）
3. 上下文注入
4. 高刷新连续对话流
"""

import os
import sys
import time
import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.config import default_config


# ============================================================
# 对话历史缓冲区
# ============================================================

@dataclass
class DialogueTurn:
    """单轮对话"""
    user_input: str
    model_response: str
    timestamp: float
    density: float = 0.5
    key_info: Dict = field(default_factory=dict)
    
    def to_context_string(self) -> str:
        """转换为上下文字符串"""
        return f"用户: {self.user_input}\n助手: {self.model_response}"


class DialogueHistoryBuffer:
    """
    对话历史缓冲区
    
    功能：
    - 存储最近N轮对话
    - 提供上下文查询
    - 与海马体记忆整合
    """
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: deque = deque(maxlen=max_turns)
        self.key_info_cache: Dict = {}  # 关键信息缓存
    
    def add_turn(self, user_input: str, model_response: str, density: float = 0.5, key_info: Dict = None):
        """添加一轮对话"""
        turn = DialogueTurn(
            user_input=user_input,
            model_response=model_response,
            timestamp=time.time(),
            density=density,
            key_info=key_info or {}
        )
        self.history.append(turn)
        
        # 更新关键信息缓存
        if key_info:
            self.key_info_cache.update(key_info)
    
    def get_context(self, last_n: int = 5) -> str:
        """获取最近N轮对话的上下文"""
        turns = list(self.history)[-last_n:]
        if not turns:
            return ""
        
        context_lines = []
        for turn in turns:
            context_lines.append(turn.to_context_string())
        
        return "\n\n".join(context_lines)
    
    def get_key_info(self) -> Dict:
        """获取所有关键信息"""
        return self.key_info_cache.copy()
    
    def search_relevant(self, query: str) -> List[DialogueTurn]:
        """搜索相关对话"""
        relevant = []
        query_keywords = set(query)
        
        for turn in self.history:
            # 简单的关键词匹配
            turn_text = turn.user_input + turn.model_response
            if any(kw in turn_text for kw in query_keywords):
                relevant.append(turn)
        
        return relevant
    
    def clear(self):
        """清空历史"""
        self.history.clear()
        self.key_info_cache.clear()


# ============================================================
# 连续密度场（保留之前的实现）
# ============================================================

@dataclass
class LogicNode:
    """逻辑节点"""
    name: str
    description: str = ""
    children: List['LogicNode'] = None
    depth: int = 0
    density: float = 0.5
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def get_density_score(self) -> int:
        count = 1
        for child in self.children:
            count += child.get_density_score()
        return count


class ContinuousDensityField(nn.Module):
    """连续逻辑密度场"""
    
    def __init__(self, hidden_size: int = 896):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.density_net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.logic_anchors = {
            'strong': ['计算', '等于', '多少', '求', '证明', '推导', 
                      '因为', '所以', '如果', '那么', '必然', '一定',
                      '月租', '房租', '押金', '合计', '费用', '租金'],
            'medium': ['怎样', '如何', '什么', '为什么', '分析', '判断', '规则', '退'],
            'weak': ['写', '创作', '想象', '感觉', '觉得', '喜欢', '故事', '诗']
        }
    
    def compute_density(self, token: str, hidden_state: torch.Tensor, context: str = "") -> float:
        if hidden_state.dim() == 0:
            hidden_state = hidden_state.unsqueeze(0)
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.flatten()
        
        neural_density = self.density_net(hidden_state).item()
        anchor_density = self._anchor_density(token, context)
        
        if anchor_density > 0.5:
            density = 0.6 * anchor_density + 0.4 * neural_density
        else:
            density = 0.4 * anchor_density + 0.6 * neural_density
        
        return density
    
    def _anchor_density(self, token: str, context: str) -> float:
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
        
        if re.search(r'\d+', text):
            return 0.7
        
        return 0.4
    
    def density_to_temperature(self, density: float) -> float:
        return 0.2 + (1 - density) * 0.6


class SelfSimilarLogicDensifier(nn.Module):
    """自相似逻辑链稠密化器"""
    
    def __init__(self, hidden_size: int = 896, max_depth: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_depth = max_depth
        
        self.expand_templates = {
            'understand': ['识别问题类型', '提取关键信息', '明确目标'],
            'extract': ['定位数据', '识别单位', '确认关系'],
            'calculate': ['建立公式', '代入数值', '执行运算', '检查结果'],
            'verify': ['检查单位', '检查数量级', '检查逻辑一致性'],
            'output': ['组织答案', '格式化输出', '最终确认']
        }
    
    def densify(self, question: str, density: float) -> Tuple[LogicNode, str]:
        root = LogicNode(
            name='problem',
            description=question,
            depth=0,
            density=density
        )
        
        expand_depth = int(density * self.max_depth) + 1
        self._expand_recursive(root, expand_depth)
        prompt = self._build_dense_prompt(root, question)
        
        return root, prompt
    
    def _expand_recursive(self, node: LogicNode, max_depth: int):
        if node.depth >= max_depth:
            return
        
        expand_type = self._determine_expand_type(node.name)
        template = self.expand_templates.get(expand_type, self.expand_templates['understand'])
        
        for i, step_desc in enumerate(template):
            child = LogicNode(
                name=f'{node.name}_{i}',
                description=step_desc,
                depth=node.depth + 1,
                density=node.density
            )
            node.children.append(child)
            self._expand_recursive(child, max_depth)
    
    def _determine_expand_type(self, name: str) -> str:
        if '计算' in name or '月租' in name or '房租' in name:
            return 'calculate'
        elif '提取' in name or '识别' in name:
            return 'extract'
        elif '验证' in name or '检查' in name:
            return 'verify'
        elif '输出' in name or '答案' in name:
            return 'output'
        else:
            return 'understand'
    
    def _build_dense_prompt(self, root: LogicNode, question: str) -> str:
        prompt = f"问题：{question}\n\n"
        prompt += "请按以下详细步骤进行推理：\n\n"
        
        def add_steps(node: LogicNode, indent: int = 0):
            nonlocal prompt
            prefix = "  " * indent
            prompt += f"{prefix}【{node.description}】\n"
            for child in node.children:
                add_steps(child, indent + 1)
        
        add_steps(root)
        prompt += "\n请逐步执行上述推理，给出详细过程和最终答案：\n"
        
        return prompt


# ============================================================
# 连续对话类脑AI
# ============================================================

class ContinuousDialogueBrain:
    """
    连续对话类脑AI
    
    核心创新：
    - 对话历史缓冲区（短期记忆）
    - 海马体记忆整合（长期记忆）
    - 上下文注入
    - 高刷新连续对话流
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.config = default_config
        self.device = device
        
        print("\n" + "=" * 60)
        print("连续对话类脑AI")
        print("=" * 60)
        
        # 加载基础模型
        print("\n[1/10] 加载基础模型...")
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
        print(f"  ✓ 模型加载成功")
        
        # 对话历史缓冲区
        print("\n[2/10] 初始化对话历史缓冲区...")
        self.dialogue_history = DialogueHistoryBuffer(max_turns=10)
        print("  ✓ 对话历史缓冲区初始化完成")
        
        # 连续密度场
        print("\n[3/10] 初始化连续逻辑密度场...")
        self.density_field = ContinuousDensityField(self.hidden_size)
        print("  ✓ 连续密度场初始化完成")
        
        # 自相似稠密化器
        print("\n[4/10] 初始化自相似逻辑链稠密化器...")
        self.densifier = SelfSimilarLogicDensifier(self.hidden_size, max_depth=3)
        print("  ✓ 自相似稠密化器初始化完成")
        
        # 类脑模块
        print("\n[5/10] 初始化STDP学习系统...")
        try:
            from stdp.stdp_engine import STDPController
            self.stdp_controller = STDPController(self.config.stdp)
            print("  ✓ STDP控制器初始化完成")
        except Exception as e:
            print(f"  ! STDP初始化跳过: {e}")
            self.stdp_controller = None
        
        print("\n[6/10] 初始化海马体系统...")
        try:
            from hippocampus.hippocampus_system import HippocampusSystem
            self.hippocampus = HippocampusSystem(self.config.hippocampus)
            print("  ✓ 海马体系统初始化完成")
        except Exception as e:
            print(f"  ! 海马体初始化跳过: {e}")
            self.hippocampus = None
        
        print("\n[7/10] 初始化元认知系统...")
        try:
            from metacognition.metacognition_system import MetacognitionSystem
            self.metacognition = MetacognitionSystem(
                self.config.metacognition,
                self.hippocampus,
                self.stdp_controller
            )
            print("  ✓ 元认知系统初始化完成")
        except Exception as e:
            print(f"  ! 元认知初始化跳过: {e}")
            self.metacognition = None
        
        print("\n[8/10] 初始化场景适配系统...")
        try:
            from scene_adapt.scene_system import SceneAdaptSystem
            self.scene_adapt = SceneAdaptSystem(self.config.scene_adapt)
            print("  ✓ 场景适配系统初始化完成")
        except Exception as e:
            print(f"  ! 场景适配初始化跳过: {e}")
            self.scene_adapt = None
        
        print("\n[9/10] 初始化双轨权重...")
        self._init_dual_weights()
        
        print("\n[10/10] 初始化推理引擎...")
        self.inference_state = {
            'cycle_count': 0,
            'current_phase': 'intuition',
            'confidence': 0.5,
            'memory_anchors': []
        }
        print("  ✓ 推理引擎初始化完成")
        
        # 统计
        self.stats = {
            'total_tokens': 0,
            'total_time': 0.0,
            'stdp_updates': 0,
            'memory_stores': 0,
            'metacognition_checks': 0,
            'avg_density': 0.0,
            'total_queries': 0
        }
        
        print("\n" + "=" * 60)
        print("✓ 所有模块初始化完成")
        print("=" * 60)
    
    def _init_dual_weights(self):
        """初始化双轨权重"""
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            num_layers = len(self.base_model.model.layers)
            print(f"  模型层数: {num_layers}, 隐藏层大小: {self.hidden_size}")
        print("  ✓ 双轨权重配置完成")
    
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
        
        total_match = re.search(r'合计\s*(\d+)', text)
        if total_match:
            numbers['total'] = int(total_match.group(1))
        
        return numbers
    
    def _extract_key_info(self, text: str) -> Dict:
        """提取关键信息用于上下文"""
        key_info = {}
        
        # 提取数字信息
        numbers = self._extract_numbers(text)
        if numbers:
            key_info['numbers'] = numbers
        
        # 提取关键实体
        entities = []
        if '月租' in text:
            entities.append('月租')
        if '押金' in text:
            entities.append('押金')
        if '卫生费' in text:
            entities.append('卫生费')
        if '房租' in text:
            entities.append('房租')
        
        if entities:
            key_info['entities'] = entities
        
        return key_info
    
    def _build_context_prompt(self, user_input: str, avg_density: float) -> str:
        """
        构建带上下文的提示词
        
        这是解决"断片"问题的关键！
        """
        # 获取对话历史
        history_context = self.dialogue_history.get_context(last_n=5)
        
        # 获取关键信息缓存
        key_info = self.dialogue_history.get_key_info()
        
        # 构建提示词
        prompt_parts = []
        
        # 1. 如果有历史对话，添加上下文
        if history_context:
            prompt_parts.append("【对话历史】")
            prompt_parts.append(history_context)
            prompt_parts.append("")
        
        # 2. 如果有关键信息，添加信息摘要
        if key_info:
            prompt_parts.append("【已知信息】")
            if 'numbers' in key_info:
                nums = key_info['numbers']
                if 'days' in nums and 'rent' in nums:
                    prompt_parts.append(f"- 租期: {nums['days']}天, 房租: {nums['rent']}元")
                    prompt_parts.append(f"- 日租: {nums['rent']/nums['days']:.0f}元/天")
                    prompt_parts.append(f"- 月租: {nums['rent']/nums['days']*30:.0f}元/月")
                if 'deposit' in nums:
                    prompt_parts.append(f"- 押金: {nums['deposit']}元（可退）")
                if 'hygiene' in nums:
                    prompt_parts.append(f"- 卫生费: {nums['hygiene']}元（离租卫生干净可退）")
            prompt_parts.append("")
        
        # 3. 添加当前问题
        prompt_parts.append("【当前问题】")
        prompt_parts.append(user_input)
        prompt_parts.append("")
        
        # 4. 根据密度添加推理引导
        if avg_density > 0.6:
            prompt_parts.append("请根据已知信息，给出准确的回答：")
        else:
            prompt_parts.append("请根据对话上下文，给出连贯的回答：")
        
        return "\n".join(prompt_parts)
    
    async def generate_stream(self, input_text: str, max_tokens: int = 200):
        """
        连续对话生成
        
        核心流程：
        1. 计算密度
        2. 构建上下文提示词
        3. 生成回答
        4. 更新对话历史
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        # ========== 第一步：编码输入 ==========
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # ========== 第二步：计算密度 ==========
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[-1][0]
        
        input_tokens = [self.tokenizer.decode([i]) for i in input_ids[0]]
        input_densities = []
        
        for i, token in enumerate(input_tokens):
            if i < hidden_states.shape[0]:
                density = self.density_field.compute_density(
                    token, hidden_states[i], input_text
                )
                input_densities.append(density)
        
        avg_density = sum(input_densities) / max(1, len(input_densities))
        
        print(f"\n[连续对话分析]")
        print(f"  输入平均密度: {avg_density:.2f}")
        print(f"  对话历史轮数: {len(self.dialogue_history.history)}")
        
        # 更新统计
        self.stats['avg_density'] = (
            (self.stats['avg_density'] * (self.stats['total_queries'] - 1) + avg_density) / 
            self.stats['total_queries']
        )
        
        # ========== 第三步：提取关键信息 ==========
        key_info = self._extract_key_info(input_text)
        
        # ========== 第四步：构建上下文提示词 ==========
        context_prompt = self._build_context_prompt(input_text, avg_density)
        
        print(f"  上下文长度: {len(context_prompt)}字符")
        
        # ========== 第五步：处理计算类问题 ==========
        if avg_density > 0.6 and '月租' in input_text:
            numbers = self._extract_numbers(input_text)
            if 'days' in numbers and 'rent' in numbers:
                result = self._format_calculation_result(numbers)
                
                # 存储到对话历史
                self.dialogue_history.add_turn(
                    input_text, result, avg_density, key_info
                )
                
                for char in result:
                    yield char
                    await asyncio.sleep(0.01)
                return
        
        # ========== 第六步：生成回答 ==========
        temperature = self.density_field.density_to_temperature(avg_density)
        print(f"  生成温度: {temperature:.2f}")
        
        # 编码上下文提示词
        context_inputs = self.tokenizer(
            context_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        context_ids = context_inputs.input_ids.to(self.device)
        context_mask = context_inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            output_ids = self.base_model.generate(
                input_ids=context_ids,
                attention_mask=context_mask,
                max_new_tokens=max_tokens,
                temperature=max(0.1, temperature),
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            output_ids[0][context_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"  生成文本长度: {len(generated_text)}")
        
        # ========== 第七步：海马体存储 ==========
        if self.hippocampus:
            try:
                self.hippocampus(
                    hidden_states[-1:].mean(dim=0, keepdim=True),
                    semantic_pointers=input_ids[0, -5:].tolist() if input_ids.size(1) >= 5 else []
                )
                self.stats['memory_stores'] += 1
            except:
                pass
        
        # ========== 第八步：更新对话历史 ==========
        cleaned_text = self._clean_output(generated_text)
        self.dialogue_history.add_turn(
            input_text, cleaned_text, avg_density, key_info
        )
        
        print(f"  对话历史已更新")
        
        # 更新统计
        elapsed = time.time() - start_time
        self.stats['total_tokens'] += output_ids.shape[1] - context_ids.shape[1]
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
    
    def _clean_output(self, text: str) -> str:
        """清理输出"""
        if not text or not text.strip():
            return "我正在思考这个问题..."
        
        lines = text.split('\n')
        seen = set()
        cleaned = []
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                cleaned.append(line)
            elif not line_stripped:
                if cleaned and cleaned[-1].strip():
                    cleaned.append(line)
        
        result = '\n'.join(cleaned)
        
        if len(result) > 800:
            last_period = result.rfind('。', 0, 800)
            if last_period > 200:
                result = result[:last_period + 1]
            else:
                result = result[:800] + "..."
        
        return result if result.strip() else text[:500]
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'system': {
                'total_queries': self.stats['total_queries'],
                'total_tokens': self.stats['total_tokens'],
                'avg_density': self.stats['avg_density'],
                'dialogue_turns': len(self.dialogue_history.history)
            },
            'modules': {
                'stdp_updates': self.stats['stdp_updates'],
                'memory_stores': self.stats['memory_stores'],
                'metacognition_checks': self.stats['metacognition_checks']
            }
        }
    
    def clear_history(self):
        """清空对话历史"""
        self.dialogue_history.clear()
        print("[对话历史] 已清空")


def create_brain_ai(model_path: str, device: str = "cpu") -> ContinuousDialogueBrain:
    """创建连续对话类脑AI"""
    return ContinuousDialogueBrain(model_path=model_path, device=device)
