"""
模块2：多尺度时序嵌套推理引擎

核心功能：
- 10ms/100Hz原生执行周期
- 三通路认知：直觉通路、逻辑推理、深度反思
- 层级化动态锚点O(1)注意力
- 时分复用架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass
import time
import asyncio


@dataclass
class InferenceState:
    """推理状态"""
    cycle_count: int = 0
    current_phase: str = "intuition"  # intuition, logic, reflection
    path_scores: Dict[int, float] = None
    generated_tokens: List[int] = None
    confidence: float = 0.5
    
    def __post_init__(self):
        if self.path_scores is None:
            self.path_scores = {}
        if self.generated_tokens is None:
            self.generated_tokens = []


class AnchorManager:
    """
    三级锚点管理器
    
    即时锚点：最近10个token
    短期锚点：1个情景记忆
    长期锚点：1个语义记忆
    """
    
    def __init__(self, config):
        self.config = config
        
        # 即时锚点缓存
        self.immediate_anchors: deque = deque(maxlen=config.immediate_anchor_count)
        
        # 短期和长期锚点
        self.short_term_anchor: Optional[str] = None
        self.short_term_feature: Optional[torch.Tensor] = None
        self.long_term_anchor: Optional[str] = None
        self.long_term_feature: Optional[torch.Tensor] = None
    
    def update_immediate(self, token_id: int, hidden_state: torch.Tensor):
        """更新即时锚点"""
        self.immediate_anchors.append({
            'token_id': token_id,
            'hidden': hidden_state.detach().clone()
        })
    
    def update_short_term(self, memory_id: str, feature: torch.Tensor):
        """更新短期锚点"""
        self.short_term_anchor = memory_id
        self.short_term_feature = feature
    
    def update_long_term(self, memory_id: str, feature: torch.Tensor):
        """更新长期锚点"""
        self.long_term_anchor = memory_id
        self.long_term_feature = feature
    
    def get_anchor_features(self) -> List[torch.Tensor]:
        """获取所有锚点特征"""
        features = []
        
        # 即时锚点
        for anchor in self.immediate_anchors:
            features.append(anchor['hidden'])
        
        # 短期锚点
        if self.short_term_feature is not None:
            features.append(self.short_term_feature)
        
        # 长期锚点
        if self.long_term_feature is not None:
            features.append(self.long_term_feature)
        
        return features
    
    def get_attention_context(
        self,
        current_hidden: torch.Tensor,
        max_anchors: int = 3
    ) -> torch.Tensor:
        """
        获取注意力上下文
        
        仅处理当前token + 3个锚点，O(1)复杂度
        """
        anchor_features = self.get_anchor_features()[-max_anchors:]
        
        if not anchor_features:
            return current_hidden
        
        # 拼接锚点特征
        context = torch.stack([current_hidden] + anchor_features, dim=1)
        
        return context


class IntuitionPathway:
    """
    直觉通路
    
    基础执行单元：10ms/周期
    负责单token生成、即时响应、特征提取
    """
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.tokens_per_cycle = config.tokens_per_cycle
    
    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        memory_anchors: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        执行一个直觉通路周期
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            memory_anchors: 记忆锚点
        
        Returns:
            output: 包含next_token, hidden_state, features等
        """
        start_time = time.time()
        
        # 模型前向传播
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                memory_anchors=memory_anchors,
                output_hidden_states=True,
                return_dict=True
            )
        
        # 获取logits和hidden states
        logits = outputs.logits[:, -1, :]
        hidden_state = outputs.hidden_states[-1][:, -1, :]
        
        # 采样下一个token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        cycle_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'next_token': next_token.item(),
            'logits': logits,
            'hidden_state': hidden_state,
            'probs': probs,
            'cycle_time_ms': cycle_time
        }


class LogicReasoningUnit:
    """
    逻辑推理单元
    
    每5个连续10ms周期为一个推理单元：
    - 周期1：生成3条推理路径
    - 周期2-3：验证角色校验
    - 周期4：辩论角色交叉纠错
    - 周期5：裁判角色最终打分
    """
    
    def __init__(self, config, model, role_adapter):
        self.config = config
        self.model = model
        self.role_adapter = role_adapter
        
        self.temperatures = config.logic_temperatures
        self.num_paths = config.logic_num_paths
    
    def generate_paths(
        self,
        input_text: str,
        max_tokens: int = 50
    ) -> List[Dict]:
        """
        周期1：生成3条独立推理路径
        
        通过不同temperature实现路径隔离
        """
        paths = []
        
        for i, temp in enumerate(self.temperatures):
            # 使用提案者角色
            prompt = self.role_adapter.format_input('proposer', input_text)
            
            # 生成路径
            path_output = self._generate_with_temp(prompt, temp, max_tokens)
            
            paths.append({
                'id': i,
                'content': path_output,
                'temperature': temp
            })
        
        return paths
    
    def validate_paths(
        self,
        paths: List[Dict]
    ) -> List[Dict]:
        """
        周期2-3：验证角色校验
        
        对每条路径做事实准确性和逻辑连贯性校验
        """
        validated_paths = []
        
        for path in paths:
            # 使用验证者角色
            prompt = self.role_adapter.format_input('validator', path['content'])
            
            # 生成验证结果
            validation = self._generate_with_temp(prompt, 0.3, 100)
            
            # 提取分数（简化实现）
            fact_score = self._extract_score(validation, "事实准确性")
            logic_score = self._extract_score(validation, "逻辑连贯性")
            
            path['validation'] = {
                'fact_score': fact_score,
                'logic_score': logic_score,
                'total_score': (fact_score + logic_score) / 2
            }
            
            validated_paths.append(path)
        
        return validated_paths
    
    def debate_paths(
        self,
        paths: List[Dict]
    ) -> List[Dict]:
        """
        周期4：辩论角色交叉纠错
        
        对打分前2的路径做交叉纠错
        """
        # 按分数排序
        sorted_paths = sorted(paths, key=lambda x: x['validation']['total_score'], reverse=True)
        top_paths = sorted_paths[:2]
        
        debated_paths = []
        
        for path in top_paths:
            # 使用辩论者角色
            prompt = self.role_adapter.format_input('debater', path['content'])
            
            # 生成辩论结果
            debate = self._generate_with_temp(prompt, 0.4, 100)
            
            path['debate'] = debate
            debated_paths.append(path)
        
        return debated_paths
    
    def judge_paths(
        self,
        paths: List[Dict]
    ) -> Tuple[Dict, Dict[int, float]]:
        """
        周期5：裁判角色最终打分
        
        输出最优路径和所有路径得分
        """
        # 使用裁判角色
        all_content = "\n\n".join([f"方案{p['id']}: {p['content']}" for p in paths])
        prompt = self.role_adapter.format_input('judge', all_content)
        
        # 生成裁决
        judgment = self._generate_with_temp(prompt, 0.2, 150)
        
        # 提取最终分数
        path_scores = {}
        for path in paths:
            path_scores[path['id']] = path['validation']['total_score']
        
        # 选择最优路径
        best_path = max(paths, key=lambda x: x['validation']['total_score'])
        
        return best_path, path_scores
    
    def _generate_with_temp(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """使用指定温度生成"""
        # 简化实现
        return f"Generated with temp={temperature}"
    
    def _extract_score(self, text: str, keyword: str) -> float:
        """从文本中提取分数"""
        # 简化实现：返回默认分数
        import re
        pattern = rf"{keyword}[：:]\s*(\d+(?:\.\d+)?)"
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
        return 7.0  # 默认分数


class ReflectionUnit:
    """
    深度反思单元
    
    每20个连续10ms周期为一个反思单元
    仅在置信度<0.6时触发
    """
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.cycles = config.reflection_unit_cycles
        self.confidence_threshold = config.reflection_confidence_threshold
    
    def should_trigger(self, confidence: float) -> bool:
        """判断是否触发深度反思"""
        return confidence < self.confidence_threshold
    
    def execute(
        self,
        input_text: str,
        low_confidence_output: str
    ) -> Dict:
        """
        执行深度反思
        
        - 元认知校验
        - 长时序因果对齐
        - 全局逻辑复盘
        """
        result = {
            'reflection_type': 'deep',
            'corrections': [],
            'final_output': low_confidence_output
        }
        
        # 如果无法校验，输出"无法确定"
        result['final_output'] = f"经过深度反思，对于该问题我无法确定准确答案。原始输出：{low_confidence_output}"
        
        return result


class O1AttentionMechanism(nn.Module):
    """
    O(1)注意力机制
    
    仅处理当前token + 3个锚点
    复杂度固定，不随序列长度增长
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.anchor_size = config.anchor_attention_size  # 4 = 1 current + 3 anchors
    
    def forward(
        self,
        query: torch.Tensor,
        anchor_keys: torch.Tensor,
        anchor_values: torch.Tensor
    ) -> torch.Tensor:
        """
        O(1)注意力计算
        
        Args:
            query: 当前token的query [batch, 1, head_dim]
            anchor_keys: 锚点keys [batch, num_anchors, head_dim]
            anchor_values: 锚点values [batch, num_anchors, head_dim]
        
        Returns:
            output: 注意力输出 [batch, 1, head_dim]
        """
        # 仅在当前token与锚点之间计算注意力
        # 复杂度: O(num_anchors) = O(4) = O(1)
        
        attention_scores = torch.matmul(query, anchor_keys.transpose(-2, -1))
        attention_scores = attention_scores / (query.size(-1) ** 0.5)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        output = torch.matmul(attention_probs, anchor_values)
        
        return output


class TemporalInferenceEngine:
    """
    时序推理引擎
    
    整合所有推理单元，实现多尺度时序嵌套推理
    """
    
    def __init__(self, config, model, role_adapter, hippocampus_system, stdp_controller):
        self.config = config
        self.model = model
        self.role_adapter = role_adapter
        self.hippocampus = hippocampus_system
        self.stdp = stdp_controller
        
        # 初始化各单元
        self.anchor_manager = AnchorManager(config)
        self.intuition = IntuitionPathway(config, model)
        self.logic_unit = LogicReasoningUnit(config, model, role_adapter)
        self.reflection_unit = ReflectionUnit(config, model)
        self.o1_attention = O1AttentionMechanism(config)
        
        # 推理状态
        self.state = InferenceState()
        self.cycle_count = 0
    
    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Generator[Dict, None, None]:
        """
        执行一个完整的推理周期
        
        Yields:
            output: 每个周期的输出
        """
        self.cycle_count += 1
        self.state.cycle_count = self.cycle_count
        
        # 1. 获取记忆锚点
        gate_signal, anchor_ids = self.hippocampus.get_memory_anchor()
        
        # 2. 更新锚点管理器
        if anchor_ids:
            # 从海马体获取锚点特征（简化）
            pass
        
        # 3. 直觉通路执行
        intuition_output = self.intuition.step(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_anchors=gate_signal
        )
        
        # 4. 更新即时锚点
        self.anchor_manager.update_immediate(
            intuition_output['next_token'],
            intuition_output['hidden_state']
        )
        
        # 5. 计算置信度
        confidence = self._compute_confidence(intuition_output)
        self.state.confidence = confidence
        
        # 6. 根据周期数决定是否执行逻辑推理
        if self.cycle_count % 5 == 0:
            # 每5个周期执行一次逻辑推理
            yield from self._execute_logic_unit(input_ids)
        elif confidence < 0.6:
            # 低置信度触发深度反思
            yield from self._execute_reflection(input_ids, intuition_output)
        else:
            # 正常输出
            yield {
                'type': 'intuition',
                'token': intuition_output['next_token'],
                'confidence': confidence,
                'cycle_time_ms': intuition_output['cycle_time_ms']
            }
    
    def _execute_logic_unit(self, input_ids: torch.Tensor) -> Generator:
        """执行逻辑推理单元"""
        # 周期1：生成路径
        paths = self.logic_unit.generate_paths("input_text")
        
        # 周期2-3：验证
        validated = self.logic_unit.validate_paths(paths)
        
        # 周期4：辩论
        debated = self.logic_unit.debate_paths(validated)
        
        # 周期5：裁决
        best_path, path_scores = self.logic_unit.judge_paths(debated)
        
        # 应用STDP
        self.stdp.apply_self_evaluation(self.model, path_scores, best_path['id'])
        
        yield {
            'type': 'logic',
            'best_path': best_path,
            'path_scores': path_scores,
            'confidence': best_path['validation']['total_score'] / 10.0
        }
    
    def _execute_reflection(self, input_ids: torch.Tensor, output: Dict) -> Generator:
        """执行深度反思"""
        reflection_result = self.reflection_unit.execute(
            "input_text",
            str(output)
        )
        
        yield {
            'type': 'reflection',
            'result': reflection_result,
            'confidence': 0.5
        }
    
    def _compute_confidence(self, output: Dict) -> float:
        """
        计算置信度
        
        confidence = 0.4*(1 - attention_entropy) + 0.3*stdp_activation + 0.3*semantic_similarity
        """
        # 注意力熵
        probs = output.get('probs', torch.ones(1, 1000))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = torch.log(torch.tensor(probs.size(-1), dtype=torch.float))
        attention_entropy_norm = (entropy / max_entropy).item()
        
        # STDP激活强度（简化）
        stdp_activation = 0.5
        
        # 语义一致性（简化）
        semantic_similarity = 0.5
        
        # 综合置信度
        confidence = (
            0.4 * (1 - attention_entropy_norm) +
            0.3 * stdp_activation +
            0.3 * semantic_similarity
        )
        
        return min(1.0, max(0.0, confidence))
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'cycle_count': self.cycle_count,
            'current_phase': self.state.current_phase,
            'confidence': self.state.confidence
        }


class StreamingGenerator:
    """
    流式生成器
    
    支持流式输出生成
    """
    
    def __init__(self, engine: TemporalInferenceEngine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """
        流式生成
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
        
        Yields:
            token_text: 生成的token文本
        """
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        generated_tokens = []
        
        for _ in range(max_tokens):
            # 执行推理周期
            for output in self.engine.step(input_ids):
                if output['type'] == 'intuition':
                    token = output['token']
                    generated_tokens.append(token)
                    
                    # 解码并yield
                    token_text = self.tokenizer.decode([token])
                    yield token_text
                    
                    # 更新输入
                    input_ids = torch.cat([
                        input_ids,
                        torch.tensor([[token]])
                    ], dim=-1)
                    
                    # 检查结束
                    if token == self.tokenizer.eos_token_id:
                        return
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        同步生成
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
        
        Returns:
            generated_text: 生成的文本
        """
        generated = []
        
        async def collect():
            async for token in self.generate_stream(prompt, max_tokens, temperature):
                generated.append(token)
        
        # 运行异步生成
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(collect())
        loop.close()
        
        return ''.join(generated)
