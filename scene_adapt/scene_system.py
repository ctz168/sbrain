"""
模块6：多任务场景自适应预适配模块

核心功能：
- 6大场景自动识别
- 场景专属预适配
- 在线场景自适应优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re


@dataclass
class SceneProfile:
    """场景配置"""
    name: str
    keywords: List[str]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_tokens: int = 200


class SceneClassifier:
    """
    场景分类器
    
    基于输入关键词、语义特征自动识别6大类场景：
    - 通用对话
    - 逻辑推理
    - 代码生成
    - 事实问答
    - 方案创作
    - 数学计算
    """
    
    def __init__(self, config):
        self.config = config
        
        # 场景关键词配置
        self.scene_keywords = config.scene_keywords
        
        # 场景配置
        self.scene_profiles = {
            'general_dialog': SceneProfile(
                name='general_dialog',
                keywords=['你好', '聊天', '闲聊', '怎么样', '什么'],
                temperature=0.8,
                max_tokens=150
            ),
            'logical_reasoning': SceneProfile(
                name='logical_reasoning',
                keywords=['推理', '逻辑', '因为', '所以', '证明', '推导', '分析', '判断'],
                temperature=0.5,
                max_tokens=300
            ),
            'code_generation': SceneProfile(
                name='code_generation',
                keywords=['代码', '编程', '函数', '算法', '实现', '程序', '写一个', 'Python', 'Java'],
                temperature=0.3,
                max_tokens=500
            ),
            'fact_qa': SceneProfile(
                name='fact_qa',
                keywords=['是什么', '什么时候', '在哪里', '谁', '多少', '事实', '历史', '定义'],
                temperature=0.4,
                max_tokens=200
            ),
            'creative_writing': SceneProfile(
                name='creative_writing',
                keywords=['写', '创作', '设计', '方案', '计划', '构思', '想象', '故事'],
                temperature=0.9,
                max_tokens=400
            ),
            'math_calculation': SceneProfile(
                name='math_calculation',
                keywords=['计算', '数学', '加减乘除', '等于', '求值', '方程', '求解', '公式'],
                temperature=0.2,
                max_tokens=100
            )
        }
        
        print("[SceneClassifier] 初始化完成，支持6大场景")
    
    def classify(self, input_text: str) -> Tuple[str, SceneProfile]:
        """
        分类输入文本
        
        Args:
            input_text: 输入文本
        
        Returns:
            scene_type: 场景类型
            profile: 场景配置
        """
        scores = {}
        
        # 检测是否是问答类问题（包含疑问词）
        is_question = any(kw in input_text for kw in ['怎样', '如何', '怎么', '为什么', '什么', '能不能', '可以吗', '吗？', '？'])
        
        # 检测数字和计算相关模式
        has_numbers = bool(re.search(r'\d+', input_text))
        has_calc_keywords = any(kw in input_text for kw in ['月租', '房租', '计算', '等于', '多少', '合计', '费用', '价格'])
        needs_calculation = any(kw in input_text for kw in ['是多少', '计算', '求', '等于多少', '多少钱'])
        
        for scene_type, keywords in self.scene_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in input_text:
                    score += 1
            scores[scene_type] = score
        
        # 如果是问答类问题，优先使用fact_qa
        if is_question and not needs_calculation:
            return 'fact_qa', self.scene_profiles['fact_qa']
        
        # 如果有数字和计算关键词，且确实需要计算，选择math_calculation
        if has_numbers and has_calc_keywords and needs_calculation:
            if scores.get('math_calculation', 0) > 0:
                return 'math_calculation', self.scene_profiles['math_calculation']
        
        # 选择得分最高的场景
        if max(scores.values()) == 0:
            # 没有匹配，使用通用对话
            best_scene = 'general_dialog'
        else:
            best_scene = max(scores, key=scores.get)
        
        return best_scene, self.scene_profiles[best_scene]
    
    def get_scene_profile(self, scene_type: str) -> SceneProfile:
        """获取场景配置"""
        return self.scene_profiles.get(scene_type, self.scene_profiles['general_dialog'])


class DynamicWeightManager:
    """
    动态权重管理器
    
    管理各场景的预适配动态权重
    """
    
    def __init__(self, config):
        self.config = config
        
        # 场景权重存储
        self.scene_weights: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # 当前场景
        self.current_scene = 'general_dialog'
    
    def load_scene_weights(self, scene_type: str, weights: Dict[str, torch.Tensor]):
        """加载场景权重"""
        self.scene_weights[scene_type] = weights
    
    def get_scene_weights(self, scene_type: str) -> Optional[Dict[str, torch.Tensor]]:
        """获取场景权重"""
        return self.scene_weights.get(scene_type)
    
    def switch_scene(self, scene_type: str, model) -> bool:
        """
        切换场景
        
        Args:
            scene_type: 目标场景
            model: 模型实例
        
        Returns:
            success: 是否成功
        """
        if scene_type not in self.scene_weights:
            return False
        
        weights = self.scene_weights[scene_type]
        
        # 应用权重到模型
        self._apply_weights(model, weights)
        
        self.current_scene = scene_type
        return True
    
    def _apply_weights(self, model, weights: Dict[str, torch.Tensor]):
        """应用权重到模型"""
        for name, weight in weights.items():
            # 遍历模型找到对应参数
            for param_name, param in model.named_parameters():
                if name in param_name and 'dynamic' in param_name:
                    with torch.no_grad():
                        param.data = weight.clone()
    
    def save_current_weights(self, model, scene_type: str):
        """保存当前场景权重"""
        weights = {}
        
        for name, param in model.named_parameters():
            if 'dynamic' in name:
                weights[name] = param.data.clone()
        
        self.scene_weights[scene_type] = weights


class SceneAdaptTrainer:
    """
    场景预适配训练器
    
    部署前一次性预训练：
    - 冻结90%静态权重
    - 仅针对10%动态分支
    - 每场景epoch≤3
    - STDP+少量监督信号
    """
    
    def __init__(self, config, model, stdp_controller):
        self.config = config
        self.model = model
        self.stdp = stdp_controller
        
        self.epochs = config.pretrain_epochs
        self.batch_size = config.pretrain_batch_size
        self.learning_rate = config.pretrain_learning_rate
    
    def train_scene(
        self,
        scene_type: str,
        train_data: List[Dict],
        epochs: int = None
    ) -> Dict:
        """
        训练场景适配
        
        Args:
            scene_type: 场景类型
            train_data: 训练数据
            epochs: 训练轮数
        
        Returns:
            result: 训练结果
        """
        epochs = epochs or self.epochs
        
        result = {
            'scene': scene_type,
            'epochs': epochs,
            'final_loss': 0.0,
            'improvement': 0.0
        }
        
        # 冻结静态权重
        self._freeze_static_weights()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch in self._get_batches(train_data):
                # 前向传播
                outputs = self._forward_batch(batch)
                
                # 计算损失
                loss = self._compute_loss(outputs, batch)
                
                # STDP更新（替代反向传播）
                self._apply_stdp_update(outputs, loss)
                
                epoch_loss += loss
            
            result['final_loss'] = epoch_loss / len(train_data)
        
        return result
    
    def _freeze_static_weights(self):
        """冻结静态权重"""
        for name, param in self.model.named_parameters():
            if 'static' in name or 'static_weight' in name:
                param.requires_grad = False
    
    def _get_batches(self, data: List[Dict]) -> List[Dict]:
        """获取批次数据"""
        batches = []
        for i in range(0, len(data), self.batch_size):
            batches.append(data[i:i + self.batch_size])
        return batches
    
    def _forward_batch(self, batch: Dict) -> Dict:
        """前向传播批次"""
        # 简化实现
        return {'logits': torch.zeros(1, 1000)}
    
    def _compute_loss(self, outputs: Dict, batch: Dict) -> float:
        """计算损失"""
        # 简化实现
        return 0.5
    
    def _apply_stdp_update(self, outputs: Dict, loss: float):
        """应用STDP更新"""
        confidence = max(0.1, 1.0 - loss)
        self.stdp.step(
            model=self.model,
            context_tokens=[],
            current_token=0,
            confidence=confidence
        )


class OnlineSceneOptimizer:
    """
    在线场景自适应优化器
    
    每个场景推理完成后，根据输出置信度优化动态分支权重
    """
    
    def __init__(self, config, stdp_controller):
        self.config = config
        self.stdp = stdp_controller
        
        # 场景优化历史
        self.optimization_history: Dict[str, List[Dict]] = {}
    
    def optimize(
        self,
        scene_type: str,
        model,
        confidence: float,
        feedback: Optional[str] = None
    ):
        """
        在线优化场景权重
        
        Args:
            scene_type: 场景类型
            model: 模型
            confidence: 置信度
            feedback: 用户反馈
        """
        # 根据置信度调整学习强度
        if confidence >= 0.9:
            # 高置信度：强化当前权重
            self.stdp.step(
                model=model,
                context_tokens=[],
                current_token=0,
                confidence=confidence
            )
        elif confidence < 0.6:
            # 低置信度：抑制当前权重
            self.stdp.step(
                model=model,
                context_tokens=[],
                current_token=0,
                confidence=confidence
            )
        
        # 记录优化历史
        if scene_type not in self.optimization_history:
            self.optimization_history[scene_type] = []
        
        self.optimization_history[scene_type].append({
            'confidence': confidence,
            'feedback': feedback
        })
    
    def get_scene_stats(self, scene_type: str) -> Dict:
        """获取场景优化统计"""
        history = self.optimization_history.get(scene_type, [])
        
        if not history:
            return {'optimization_count': 0}
        
        confidences = [h['confidence'] for h in history]
        
        return {
            'optimization_count': len(history),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences)
        }


class SceneAdaptSystem:
    """
    完整的场景自适应系统
    
    整合：
    - 场景分类
    - 权重管理
    - 预适配训练
    - 在线优化
    """
    
    def __init__(self, config, model=None, stdp_controller=None):
        self.config = config
        
        # 初始化子模块
        self.classifier = SceneClassifier(config)
        self.weight_manager = DynamicWeightManager(config)
        
        if model and stdp_controller:
            self.trainer = SceneAdaptTrainer(config, model, stdp_controller)
            self.optimizer = OnlineSceneOptimizer(config, stdp_controller)
        else:
            self.trainer = None
            self.optimizer = None
        
        self.current_scene = 'general_dialog'
        self.current_profile = self.classifier.get_scene_profile('general_dialog')
        
        print("[SceneAdapt] 场景自适应系统初始化完成")
    
    def process(
        self,
        input_text: str,
        model=None
    ) -> Tuple[str, SceneProfile]:
        """
        处理输入，识别场景并切换
        
        Args:
            input_text: 输入文本
            model: 模型实例
        
        Returns:
            scene_type: 场景类型
            profile: 场景配置
        """
        # 1. 分类场景
        scene_type, profile = self.classifier.classify(input_text)
        
        # 2. 切换场景权重
        if scene_type != self.current_scene and model:
            self.weight_manager.switch_scene(scene_type, model)
        
        self.current_scene = scene_type
        self.current_profile = profile
        
        return scene_type, profile
    
    def update_after_inference(
        self,
        model,
        confidence: float,
        feedback: str = None
    ):
        """
        推理后更新
        
        Args:
            model: 模型
            confidence: 置信度
            feedback: 用户反馈
        """
        if self.optimizer:
            self.optimizer.optimize(
                self.current_scene,
                model,
                confidence,
                feedback
            )
    
    def pretrain_all_scenes(
        self,
        model,
        train_data: Dict[str, List[Dict]]
    ) -> Dict:
        """
        预训练所有场景
        
        Args:
            model: 模型
            train_data: 各场景训练数据
        
        Returns:
            results: 训练结果
        """
        if not self.trainer:
            return {}
        
        results = {}
        
        for scene_type, data in train_data.items():
            result = self.trainer.train_scene(scene_type, data)
            results[scene_type] = result
            
            # 保存训练后的权重
            self.weight_manager.save_current_weights(model, scene_type)
        
        return results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            'current_scene': self.current_scene,
            'available_scenes': list(self.classifier.scene_profiles.keys()),
            'loaded_weights': list(self.weight_manager.scene_weights.keys())
        }
        
        if self.optimizer:
            stats['optimization_stats'] = {
                scene: self.optimizer.get_scene_stats(scene)
                for scene in self.classifier.scene_profiles.keys()
            }
        
        return stats
