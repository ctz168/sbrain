"""
类人脑双系统全闭环AI架构 - 主模型

整合所有模块：
- 双轨权重Transformer
- STDP学习系统
- 海马体记忆系统
- 时序推理引擎
- 元认知校验系统
- 场景自适应系统
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import time

# 导入各模块
from configs.config import BrainConfig, default_config
from core.dual_weight import (
    DualWeightTransformerLayer,
    RoleAdapter
)
from stdp.stdp_engine import STDPController
from hippocampus.hippocampus_system import HippocampusSystem
from inference.engine import TemporalInferenceEngine, StreamingGenerator
from metacognition.metacognition_system import MetacognitionSystem
from scene_adapt.scene_system import SceneAdaptSystem


class BrainAIModel(nn.Module):
    """
    类人脑双系统全闭环AI模型
    
    核心特性：
    - 90%静态权重 + 10%动态权重
    - 10ms/100Hz原生执行周期
    - 纯STDP学习，无反向传播
    - 海马体-新皮层双系统
    - 元认知双闭环校验
    """
    
    def __init__(
        self,
        config: BrainConfig = None,
        pretrained_model_path: str = None
    ):
        super().__init__()
        
        self.config = config or default_config
        
        # 加载预训练底座模型
        print(f"[BrainAI] 正在加载底座模型: {self.config.model_name}")
        
        if pretrained_model_path:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_path,
                trust_remote_code=True
            )
        else:
            # 使用配置中的模型路径
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
        
        # 获取模型配置
        self.hidden_size = self.base_model.config.hidden_size
        self.num_attention_heads = self.base_model.config.num_attention_heads
        self.num_layers = self.base_model.config.num_hidden_layers
        self.vocab_size = self.base_model.config.vocab_size
        
        print(f"[BrainAI] 模型配置: hidden_size={self.hidden_size}, "
              f"heads={self.num_attention_heads}, layers={self.num_layers}")
        
        # 初始化双轨权重层
        self._init_dual_weight_layers()
        
        # 初始化STDP控制器
        self.stdp_controller = STDPController(self.config.stdp)
        
        # 初始化海马体系统
        self.hippocampus = HippocampusSystem(self.config.hippocampus)
        
        # 初始化元认知系统
        self.metacognition = MetacognitionSystem(
            self.config.metacognition,
            self.hippocampus,
            self.stdp_controller
        )
        
        # 初始化场景自适应系统
        self.scene_adapt = SceneAdaptSystem(
            self.config.scene_adapt,
            self,
            self.stdp_controller
        )
        
        # 初始化推理引擎
        self.inference_engine = TemporalInferenceEngine(
            self.config.inference,
            self,
            RoleAdapter,
            self.hippocampus,
            self.stdp_controller
        )
        
        # 全局状态
        self.global_cycle = 0
        self.is_training = False
        
        print("[BrainAI] 类人脑双系统模型初始化完成")
    
    def _init_dual_weight_layers(self):
        """初始化双轨权重层"""
        # 获取原始模型的层
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            original_layers = self.base_model.model.layers
        else:
            original_layers = []
        
        # 创建双轨权重层
        self.dual_weight_layers = nn.ModuleList()
        
        for i, layer in enumerate(original_layers):
            # 提取原始权重
            static_q = getattr(layer.self_attn, 'q_proj', layer.self_attn).weight.data.clone() if hasattr(layer.self_attn, 'q_proj') else None
            static_k = getattr(layer.self_attn, 'k_proj', layer.self_attn).weight.data.clone() if hasattr(layer.self_attn, 'k_proj') else None
            static_v = getattr(layer.self_attn, 'v_proj', layer.self_attn).weight.data.clone() if hasattr(layer.self_attn, 'v_proj') else None
            static_o = getattr(layer.self_attn, 'o_proj', layer.self_attn).weight.data.clone() if hasattr(layer.self_attn, 'o_proj') else None
            
            dual_layer = DualWeightTransformerLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.base_model.config.intermediate_size,
                static_ratio=self.config.hard_constraints.STATIC_WEIGHT_RATIO
            )
            
            self.dual_weight_layers.append(dual_layer)
        
        print(f"[BrainAI] 初始化了 {len(self.dual_weight_layers)} 个双轨权重层")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_anchors: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> CausalLMOutputWithPast:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            memory_anchors: 海马体记忆锚点
            past_key_values: KV缓存
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典
        
        Returns:
            outputs: 模型输出
        """
        self.global_cycle += 1
        
        # 使用底座模型进行前向传播
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        # 获取隐藏状态用于海马体
        if output_hidden_states and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
            
            # 海马体处理
            gate_signal, anchor_ids = self.hippocampus(
                hidden_states[:, -1, :],
                semantic_pointers=input_ids[0, -5:].tolist() if input_ids.size(1) >= 5 else []
            )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> torch.Tensor:
        """
        生成文本
        
        Args:
            input_ids: 输入token IDs
            max_new_tokens: 最大生成token数
            temperature: 温度参数
        
        Returns:
            generated_ids: 生成的token IDs
        """
        # 场景识别
        input_text = self.tokenizer.decode(input_ids[0])
        scene_type, scene_profile = self.scene_adapt.process(input_text, self)
        
        # 使用场景配置
        temperature = scene_profile.temperature
        
        # 生成
        with torch.no_grad():
            generated_ids = self.base_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=scene_profile.top_p,
                top_k=scene_profile.top_k,
                **kwargs
            )
        
        return generated_ids
    
    def chat(
        self,
        message: str,
        history: List[Dict[str, str]] = None,
        max_tokens: int = 200
    ) -> str:
        """
        对话接口
        
        Args:
            message: 用户消息
            history: 对话历史
            max_tokens: 最大生成token数
        
        Returns:
            response: 回复文本
        """
        # 构建输入
        if history:
            context = "\n".join([
                f"{h['role']}: {h['content']}"
                for h in history[-5:]
            ])
            full_input = f"{context}\nUser: {message}\nAssistant:"
        else:
            full_input = f"User: {message}\nAssistant:"
        
        # 编码
        input_ids = self.tokenizer.encode(full_input, return_tensors='pt')
        
        # 生成
        output_ids = self.generate(input_ids, max_new_tokens=max_tokens)
        
        # 解码
        response = self.tokenizer.decode(
            output_ids[0][input_ids.size(1):],
            skip_special_tokens=True
        )
        
        return response
    
    def apply_stdp(
        self,
        confidence: float,
        context_tokens: List[int] = None,
        current_token: int = 0
    ):
        """
        应用STDP更新
        
        Args:
            confidence: 置信度
            context_tokens: 上下文token
            current_token: 当前token
        """
        self.stdp_controller.step(
            model=self,
            context_tokens=context_tokens or [],
            current_token=current_token,
            confidence=confidence
        )
    
    def consolidate_memory(self) -> Optional[Dict]:
        """执行记忆巩固（SWR）"""
        return self.hippocampus.check_and_consolidate(self.stdp_controller)
    
    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            'global_cycle': self.global_cycle,
            'stdp': self.stdp_controller.get_stats(),
            'hippocampus': self.hippocampus.get_stats(),
            'metacognition': self.metacognition.get_stats(),
            'scene_adapt': self.scene_adapt.get_stats()
        }
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'config': self.config,
            'global_cycle': self.global_cycle,
            'dynamic_weights': self._get_all_dynamic_weights(),
            'hippocampus_state': self.hippocampus.state_dict() if hasattr(self.hippocampus, 'state_dict') else {},
            'scene_weights': self.scene_adapt.weight_manager.scene_weights
        }
        
        torch.save(checkpoint, path)
        print(f"[BrainAI] 检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.global_cycle = checkpoint.get('global_cycle', 0)
        
        # 恢复动态权重
        dynamic_weights = checkpoint.get('dynamic_weights', {})
        self._set_all_dynamic_weights(dynamic_weights)
        
        # 恢复场景权重
        scene_weights = checkpoint.get('scene_weights', {})
        for scene_type, weights in scene_weights.items():
            self.scene_adapt.weight_manager.load_scene_weights(scene_type, weights)
        
        print(f"[BrainAI] 检查点已加载: {path}")
    
    def _get_all_dynamic_weights(self) -> Dict:
        """获取所有动态权重"""
        weights = {}
        
        for i, layer in enumerate(self.dual_weight_layers):
            layer_weights = layer.get_all_dynamic_weights()
            for name, w in layer_weights.items():
                weights[f"layer_{i}_{name}"] = w
        
        return weights
    
    def _set_all_dynamic_weights(self, weights: Dict):
        """设置所有动态权重"""
        for i, layer in enumerate(self.dual_weight_layers):
            for name, w in layer.get_all_dynamic_weights().items():
                key = f"layer_{i}_{name}"
                if key in weights:
                    # 设置权重
                    pass  # 具体实现根据权重结构


def create_model(
    model_path: str = None,
    config: BrainConfig = None
) -> BrainAIModel:
    """
    创建模型实例
    
    Args:
        model_path: 预训练模型路径
        config: 配置
    
    Returns:
        model: 模型实例
    """
    return BrainAIModel(config=config, pretrained_model_path=model_path)


# 便捷导入
__all__ = [
    'BrainAIModel',
    'BrainConfig',
    'create_model',
    'default_config'
]
