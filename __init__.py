"""
类人脑双系统全闭环AI架构

基于Qwen3.5-0.8B底座模型
"""

from .core.model import BrainAIModel, create_model
from .configs.config import BrainConfig, default_config
from .core.dual_weight import (
    DualWeightLinear,
    DualWeightAttention,
    DualWeightFFN,
    DualWeightTransformerLayer,
    RoleAdapter
)
from .stdp.stdp_engine import STDPController
from .hippocampus.hippocampus_system import HippocampusSystem
from .inference.engine import TemporalInferenceEngine, StreamingGenerator
from .metacognition.metacognition_system import MetacognitionSystem
from .scene_adapt.scene_system import SceneAdaptSystem
from .evaluation.evaluator import EvaluationSystem

__version__ = '1.0.0'
__author__ = 'BrainAI Team'

__all__ = [
    # 主模型
    'BrainAIModel',
    'create_model',
    
    # 配置
    'BrainConfig',
    'default_config',
    
    # 双轨权重
    'DualWeightLinear',
    'DualWeightAttention',
    'DualWeightFFN',
    'DualWeightTransformerLayer',
    'RoleAdapter',
    
    # STDP
    'STDPController',
    
    # 海马体
    'HippocampusSystem',
    
    # 推理引擎
    'TemporalInferenceEngine',
    'StreamingGenerator',
    
    # 元认知
    'MetacognitionSystem',
    
    # 场景适配
    'SceneAdaptSystem',
    
    # 评估
    'EvaluationSystem'
]
