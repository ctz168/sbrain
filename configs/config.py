"""
类人脑双系统全闭环AI架构 - 全局配置文件
基于Qwen3.5-0.8B底座模型

严格遵循刚性红线约束：
- 90%静态权重 + 10%动态权重
- INT4量化后显存≤420MB
- 10ms/100Hz原生执行周期
- 纯STDP学习，无反向传播
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import torch


# ==================== 刚性约束配置 ====================

@dataclass
class HardConstraints:
    """不可突破的硬性约束配置"""
    # 权重安全约束
    STATIC_WEIGHT_RATIO: float = 0.9  # 90%静态基础权重
    DYNAMIC_WEIGHT_RATIO: float = 0.1  # 10%动态增量权重
    
    # 端侧算力约束
    MAX_MEMORY_MB: int = 420  # INT4量化后最大显存
    REFRESH_PERIOD_MS: int = 10  # 10ms刷新周期
    MAX_COMPUTE_OVERHEAD: float = 0.1  # 单周期算力开销≤10%
    
    # 窄窗口约束
    NARROW_WINDOW_SIZE: int = 2  # 每周期处理1-2个token
    ATTENTION_COMPLEXITY: str = "O(1)"  # 固定O(1)复杂度
    
    # 海马体内存约束
    HIPPOCAMPUS_MAX_MEMORY_KB: int = 512  # 512KB


# ==================== STDP超参数配置 ====================

@dataclass
class STDPConfig:
    """STDP时序可塑性权重更新配置"""
    # 核心学习率
    alpha_LTP: float = 0.005  # 长期增强学习率
    beta_LTD: float = 0.01  # 长期抑制学习率
    
    # 权重边界
    weight_min: float = -0.1  # 权重下界
    weight_max: float = 0.1  # 权重上界
    
    # 时序阈值
    t_pre_threshold: float = 0.5
    t_post_threshold: float = 0.5
    
    # 动态学习率适配
    high_confidence_boost: float = 1.5  # 置信度≥0.9时α乘以此值
    low_confidence_boost: float = 2.0  # 置信度<0.6时β乘以此值
    
    # 更新开关
    update_attention: bool = True
    update_ffn: bool = True
    update_hippocampus_gate: bool = True
    update_self_eval: bool = True


# ==================== 海马体系统配置 ====================

@dataclass
class HippocampusConfig:
    """海马体记忆系统配置"""
    # 内嗅皮层EC编码
    EC_input_dim: int = 768  # Qwen3.5-0.8B隐藏层维度
    EC_output_dim: int = 64  # 压缩后的稀疏特征维度
    EC_sparsity: float = 0.75  # 75%稀疏度
    
    # 齿状回DG模式分离
    DG_orthogonalization: bool = True
    DG_random_seed: int = 42  # 固定随机种子
    
    # CA3情景记忆库
    CA3_max_capacity: int = 1024  # 最大记忆条目数
    CA3_entry_size_bytes: int = 128  # 单条目大小
    CA3_timestamp_precision_ms: int = 10  # 时间戳精度
    
    # CA1时序门控
    CA1_recall_topk: int = 2  # 每周期召回1-2个锚点
    CA1_gate_base: float = 0.7
    CA1_gate_confidence_weight: float = 0.3
    
    # 长期语义记忆
    semantic_max_capacity: int = 4096  # 最大三元组数
    semantic_entry_size_bytes: int = 64
    
    # 尖波涟漪SWR
    SWR_idle_threshold_s: int = 30  # 空闲30秒触发
    SWR_replay_topk: int = 30  # 回放Top30记忆
    SWR_max_duration_s: int = 300  # 最大回放5分钟
    SWR_cleanup_visit_threshold: int = 2
    SWR_cleanup_confidence_threshold: float = 0.3


# ==================== 时序推理引擎配置 ====================

@dataclass
class InferenceConfig:
    """多尺度时序嵌套推理引擎配置"""
    # 基础执行单元
    base_cycle_ms: int = 10  # 10ms基础周期
    tokens_per_cycle: int = 2  # 每周期处理token数
    
    # 逻辑推理单元（5个周期）
    logic_unit_cycles: int = 5
    logic_temperatures: Tuple[float, ...] = (0.6, 0.7, 0.8)  # 3条推理路径
    logic_num_paths: int = 3
    
    # 深度反思单元（20个周期）
    reflection_unit_cycles: int = 20
    reflection_confidence_threshold: float = 0.6
    
    # 三级锚点体系
    immediate_anchor_count: int = 10  # 即时锚点：最近10个token
    short_term_anchor_count: int = 1  # 短期锚点：1个情景记忆
    long_term_anchor_count: int = 1  # 长期锚点：1个语义记忆
    
    # 注意力计算
    anchor_attention_size: int = 4  # 当前token + 3个锚点


# ==================== 元认知校验配置 ====================

@dataclass
class MetacognitionConfig:
    """元认知双闭环校验系统配置"""
    # 置信度计算权重
    attention_entropy_weight: float = 0.4
    stdp_activation_weight: float = 0.3
    semantic_similarity_weight: float = 0.3
    
    # 校验阈值
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.6
    
    # 校验行为
    cross_validation_expansion: int = 1  # 扩展1个推理单元
    deep_reflection_expansion: int = 2  # 扩展2个推理单元


# ==================== 场景适配配置 ====================

@dataclass
class SceneAdaptConfig:
    """多任务场景自适应配置"""
    # 场景类型
    scene_types: List[str] = field(default_factory=lambda: [
        "general_dialog",  # 通用对话
        "logical_reasoning",  # 逻辑推理
        "code_generation",  # 代码生成
        "fact_qa",  # 事实问答
        "creative_writing",  # 方案创作
        "math_calculation"  # 数学计算
    ])
    
    # 预训练参数
    pretrain_epochs: int = 3
    pretrain_batch_size: int = 8
    pretrain_learning_rate: float = 1e-5
    
    # 场景关键词
    scene_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "logical_reasoning": ["推理", "逻辑", "因为", "所以", "证明", "推导"],
        "code_generation": ["代码", "编程", "函数", "算法", "实现", "程序"],
        "fact_qa": ["是什么", "什么时候", "在哪里", "谁", "多少", "事实"],
        "creative_writing": ["写", "创作", "设计", "方案", "计划", "构思"],
        "math_calculation": ["计算", "数学", "加减乘除", "等于", "求值", "方程"]
    })


# ==================== 测评配置 ====================

@dataclass
class EvaluationConfig:
    """多维度全链路测评配置"""
    # 测评权重
    memory_weight: float = 0.4
    reasoning_weight: float = 0.3
    reliability_weight: float = 0.15
    performance_weight: float = 0.1
    learning_weight: float = 0.05
    
    # 记忆能力指标
    memory_retention_threshold: float = 0.95  # 100k token保持率≥95%
    memory_confusion_threshold: float = 0.01  # 混淆率≤1%
    cross_session_recall_threshold: float = 0.90  # 跨会话召回≥90%
    anti_forgetting_threshold: float = 0.99  # 抗遗忘≥99%
    
    # 推理能力指标
    reasoning_improvement_ratio: float = 3.0  # 较原生提升≥300%
    
    # 可靠性指标
    fact_accuracy_threshold: float = 0.90  # 事实准确率≥90%
    hallucination_ratio_threshold: float = 0.08  # 幻觉率≤8%
    
    # 端侧性能指标
    max_memory_mb: int = 420
    max_latency_ms: int = 20
    
    # 学习能力指标
    learning_speed_improvement: float = 4.0  # 提升≥400%


# ==================== 全局配置 ====================

@dataclass
class BrainConfig:
    """类人脑双系统全闭环AI架构全局配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-0.5B"  # 使用可下载的模型
    model_path: str = "./models/Qwen2.5-0.5B"
    hidden_size: int = 896  # Qwen2.5-0.5B的隐藏层维度
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    intermediate_size: int = 4864
    vocab_size: int = 151936
    
    # 子配置
    hard_constraints: HardConstraints = field(default_factory=HardConstraints)
    stdp: STDPConfig = field(default_factory=STDPConfig)
    hippocampus: HippocampusConfig = field(default_factory=HippocampusConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    metacognition: MetacognitionConfig = field(default_factory=MetacognitionConfig)
    scene_adapt: SceneAdaptConfig = field(default_factory=SceneAdaptConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # 设备配置
    device: str = "cpu"
    quantization: str = "FP32"  # FP32, INT4, INT8
    
    # 随机种子
    seed: int = 42
    
    def set_seed(self):
        """设置全局随机种子"""
        import numpy as np
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)


# 默认配置实例
default_config = BrainConfig()
