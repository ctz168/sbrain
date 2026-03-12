"""核心模块"""
from .dual_weight import (
    DualWeightLinear,
    DualWeightAttention,
    DualWeightFFN,
    DualWeightTransformerLayer,
    RoleAdapter
)
from .model import BrainAIModel, create_model

__all__ = [
    'DualWeightLinear',
    'DualWeightAttention',
    'DualWeightFFN',
    'DualWeightTransformerLayer',
    'RoleAdapter',
    'BrainAIModel',
    'create_model'
]
