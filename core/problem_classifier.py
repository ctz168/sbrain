#!/usr/bin/env python3
"""
问题类型识别与路由系统

核心功能：
1. 识别问题类型（逻辑型/创作型/混合型）
2. 计算逻辑锚点强度
3. 动态路由到不同的处理模块
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ProblemType(Enum):
    """问题类型枚举"""
    PURE_LOGIC = "pure_logic"        # 纯逻辑型
    LOGIC_DOMINANT = "logic_dominant"  # 逻辑主导型
    HYBRID = "hybrid"                # 混合型
    CREATIVE_DOMINANT = "creative_dominant"  # 创意主导型
    PURE_CREATIVE = "pure_creative"  # 纯创作型


@dataclass
class ProblemAnalysis:
    """问题分析结果"""
    problem_type: ProblemType
    logic_anchor_score: float      # 逻辑锚点强度 [0, 1]
    creative_score: float          # 创意需求强度 [0, 1]
    key_indicators: List[str]      # 关键指标
    suggested_approach: str        # 建议的处理方式
    confidence: float              # 分析置信度


class ProblemTypeClassifier:
    """
    问题类型分类器
    
    核心思想：
    1. 分析问题的语言特征
    2. 计算逻辑锚点强度
    3. 评估创意需求
    4. 动态路由
    """
    
    def __init__(self):
        # 逻辑锚点指示词
        self.logic_indicators = {
            'calculation': [
                '计算', '等于', '多少', '求值', '公式', '数学',
                '加减乘除', '百分比', '比例', '平均', '总和'
            ],
            'reasoning': [
                '因为', '所以', '如果', '那么', '推导', '证明',
                '判断', '推理', '逻辑', '因果', '必然', '一定'
            ],
            'factual': [
                '是什么', '什么时候', '在哪里', '谁', '多少',
                '怎样', '如何', '规则', '条件', '要求', '标准'
            ],
            'verification': [
                '正确', '错误', '对不对', '是否', '验证', '确认',
                '检查', '判断', '比较', '区别'
            ]
        }
        
        # 创意指示词
        self.creative_indicators = {
            'writing': [
                '写', '创作', '编', '构思', '想象', '设计',
                '故事', '小说', '诗歌', '文章', '剧本'
            ],
            'artistic': [
                '画', '描绘', '表现', '风格', '美感', '艺术',
                '创意', '灵感', '独特', '新颖'
            ],
            'open_ended': [
                '你觉得', '你认为', '可能', '也许', '或者',
                '有什么想法', '怎么看待', '分享', '谈谈'
            ],
            'exploratory': [
                '探索', '尝试', '实验', '创新', '突破',
                '可能性', '想象一下', '假如'
            ]
        }
        
        # 混合型指示词
        self.hybrid_indicators = [
            '分析', '评价', '建议', '方案', '规划',
            '设计一个', '写一篇分析', '谈谈你的看法'
        ]
    
    def classify(self, question: str) -> ProblemAnalysis:
        """
        分类问题类型
        
        Args:
            question: 输入问题
        
        Returns:
            ProblemAnalysis: 分析结果
        """
        # 1. 计算逻辑锚点强度
        logic_score, logic_indicators = self._calculate_logic_score(question)
        
        # 2. 计算创意需求强度
        creative_score, creative_indicators = self._calculate_creative_score(question)
        
        # 3. 确定问题类型
        problem_type = self._determine_type(logic_score, creative_score)
        
        # 4. 确定处理方式
        approach = self._suggest_approach(problem_type, logic_score, creative_score)
        
        # 5. 计算置信度
        confidence = self._calculate_confidence(logic_score, creative_score)
        
        return ProblemAnalysis(
            problem_type=problem_type,
            logic_anchor_score=logic_score,
            creative_score=creative_score,
            key_indicators=logic_indicators + creative_indicators,
            suggested_approach=approach,
            confidence=confidence
        )
    
    def _calculate_logic_score(self, question: str) -> Tuple[float, List[str]]:
        """计算逻辑锚点强度"""
        score = 0.0
        indicators = []
        
        # 检查各类逻辑指示词
        for category, words in self.logic_indicators.items():
            for word in words:
                if word in question:
                    score += 0.15
                    indicators.append(f"[{category}]{word}")
        
        # 检查数字存在
        if re.search(r'\d+', question):
            score += 0.2
            indicators.append("[numeric]存在数字")
        
        # 检查数学运算符
        if re.search(r'[+\-×÷=<>]', question):
            score += 0.15
            indicators.append("[operator]数学运算符")
        
        # 检查疑问句式
        if '？' in question or '?' in question:
            # 具体问题加分
            if any(w in question for w in ['多少', '是什么', '怎样']):
                score += 0.1
        
        # 限制在 [0, 1]
        score = min(1.0, score)
        
        return score, indicators
    
    def _calculate_creative_score(self, question: str) -> Tuple[float, List[str]]:
        """计算创意需求强度"""
        score = 0.0
        indicators = []
        
        # 检查各类创意指示词
        for category, words in self.creative_indicators.items():
            for word in words:
                if word in question:
                    score += 0.12
                    indicators.append(f"[{category}]{word}")
        
        # 检查开放性问题
        open_patterns = [
            r'你觉得.*怎样',
            r'你认为.*应该',
            r'写.*关于',
            r'设计.*方案',
            r'谈谈.*看法'
        ]
        for pattern in open_patterns:
            if re.search(pattern, question):
                score += 0.15
                indicators.append("[open]开放性问题")
        
        # 检查主观表达
        subjective_words = ['感觉', '觉得', '认为', '想法', '观点', '看法']
        if any(w in question for w in subjective_words):
            score += 0.1
            indicators.append("[subjective]主观表达")
        
        # 限制在 [0, 1]
        score = min(1.0, score)
        
        return score, indicators
    
    def _determine_type(
        self,
        logic_score: float,
        creative_score: float
    ) -> ProblemType:
        """确定问题类型"""
        # 计算差异
        diff = logic_score - creative_score
        
        if diff > 0.5:
            return ProblemType.PURE_LOGIC
        elif diff > 0.2:
            return ProblemType.LOGIC_DOMINANT
        elif diff > -0.2:
            return ProblemType.HYBRID
        elif diff > -0.5:
            return ProblemType.CREATIVE_DOMINANT
        else:
            return ProblemType.PURE_CREATIVE
    
    def _suggest_approach(
        self,
        problem_type: ProblemType,
        logic_score: float,
        creative_score: float
    ) -> str:
        """建议处理方式"""
        approaches = {
            ProblemType.PURE_LOGIC: "逻辑链稠密化 + 符号计算",
            ProblemType.LOGIC_DOMINANT: "逻辑稠密化为主 + 适度创意",
            ProblemType.HYBRID: "逻辑+创意双轨处理",
            ProblemType.CREATIVE_DOMINANT: "创意激发为主 + 逻辑约束",
            ProblemType.PURE_CREATIVE: "发散思维 + 联想扩展"
        }
        return approaches[problem_type]
    
    def _calculate_confidence(
        self,
        logic_score: float,
        creative_score: float
    ) -> float:
        """计算分析置信度"""
        # 如果两个分数差异明显，置信度高
        diff = abs(logic_score - creative_score)
        
        # 如果两个分数都很低或都很高，置信度较低
        total = logic_score + creative_score
        
        if total < 0.2:
            return 0.5  # 信息不足
        elif diff > 0.3:
            return 0.9  # 类型明确
        elif diff > 0.15:
            return 0.75  # 类型较明确
        else:
            return 0.6  # 类型模糊


class DualTrackProcessor:
    """
    双轨处理器：逻辑轨 + 创意轨
    
    核心思想：
    1. 逻辑轨：处理逻辑锚点强的部分
    2. 创意轨：处理创意需求强的部分
    3. 动态融合：根据问题类型调整比例
    """
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.classifier = ProblemTypeClassifier()
    
    def process(self, question: str) -> Tuple[str, ProblemAnalysis]:
        """
        处理问题
        
        根据问题类型选择处理策略
        """
        # 1. 分析问题类型
        analysis = self.classifier.classify(question)
        
        print(f"\n[问题分析]")
        print(f"  类型: {analysis.problem_type.value}")
        print(f"  逻辑锚点: {analysis.logic_anchor_score:.2f}")
        print(f"  创意需求: {analysis.creative_score:.2f}")
        print(f"  建议方式: {analysis.suggested_approach}")
        
        # 2. 根据类型选择处理策略
        if analysis.problem_type in [ProblemType.PURE_LOGIC, ProblemType.LOGIC_DOMINANT]:
            # 逻辑轨处理
            result = self._logic_track(question, analysis)
        elif analysis.problem_type == ProblemType.PURE_CREATIVE:
            # 创意轨处理
            result = self._creative_track(question, analysis)
        else:
            # 混合处理
            result = self._hybrid_track(question, analysis)
        
        return result, analysis
    
    def _logic_track(self, question: str, analysis: ProblemAnalysis) -> str:
        """
        逻辑轨处理
        
        特点：
        - 逻辑链稠密化
        - 符号计算辅助
        - 验证每一步
        """
        # 构建逻辑导向的提示
        prompt = f"""问题：{question}

请按以下逻辑步骤分析：

【第一步：理解问题】
- 问题的核心是什么？
- 需要什么信息？

【第二步：提取关键信息】
- 有哪些关键数据/条件？
- 它们之间的关系是什么？

【第三步：建立推理链】
- 需要哪些推理步骤？
- 每步的依据是什么？

【第四步：执行推理】
- 逐步执行推理
- 验证每步的正确性

【第五步：得出结论】
- 最终答案是什么？
- 如何验证答案正确？

请给出详细的分析过程：
"""
        return self._generate(prompt)
    
    def _creative_track(self, question: str, analysis: ProblemAnalysis) -> str:
        """
        创意轨处理
        
        特点：
        - 发散思维
        - 联想扩展
        - 多角度探索
        """
        # 构建创意导向的提示
        prompt = f"""问题：{question}

请从以下角度展开思考：

【角度一：核心主题】
- 这个问题的核心是什么？
- 有哪些可以深入的方向？

【角度二：联想扩展】
- 有哪些相关的概念/想法？
- 可以如何延伸？

【角度三：创意表达】
- 如何用独特的方式呈现？
- 有什么新颖的视角？

【角度四：情感共鸣】
- 如何让内容更有感染力？
- 如何与读者建立连接？

请自由发挥，展现创意：
"""
        return self._generate(prompt, temperature=0.8)
    
    def _hybrid_track(self, question: str, analysis: ProblemAnalysis) -> str:
        """
        混合轨处理
        
        特点：
        - 逻辑部分用逻辑轨
        - 创意部分用创意轨
        - 动态融合
        """
        # 计算逻辑和创意的比例
        logic_ratio = analysis.logic_anchor_score / (
            analysis.logic_anchor_score + analysis.creative_score + 0.01
        )
        creative_ratio = 1 - logic_ratio
        
        # 构建混合提示
        prompt = f"""问题：{question}

请结合逻辑分析和创意思考来回答：

【逻辑分析部分】({logic_ratio*100:.0f}%权重)
- 事实依据是什么？
- 有哪些逻辑推理？

【创意思考部分】({creative_ratio*100:.0f}%权重)
- 有什么独特的视角？
- 如何让内容更有价值？

【综合输出】
请结合以上两部分，给出完整的回答：
"""
        return self._generate(prompt, temperature=0.5)
    
    def _generate(self, prompt: str, temperature: float = 0.3) -> str:
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=temperature,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# 测试用例
if __name__ == "__main__":
    classifier = ProblemTypeClassifier()
    
    test_cases = [
        # 纯逻辑型
        "20天房租1600元，月租是多少？",
        "如果A>B，B>C，那么A和C的关系是什么？",
        "押金怎样才能退？",
        
        # 逻辑主导型
        "分析一下人工智能的发展趋势",
        "如何评价这个方案的可行性？",
        
        # 混合型
        "写一篇关于人工智能发展前景的分析文章",
        "设计一个智能家居系统方案",
        
        # 创意主导型
        "写一首关于春天的诗",
        "设计一个有创意的产品",
        
        # 纯创作型
        "讲一个有趣的故事",
        "想象一下未来的世界是什么样子"
    ]
    
    print("=" * 60)
    print("问题类型识别测试")
    print("=" * 60)
    
    for question in test_cases:
        analysis = classifier.classify(question)
        print(f"\n问题: {question[:30]}...")
        print(f"  类型: {analysis.problem_type.value}")
        print(f"  逻辑锚点: {analysis.logic_anchor_score:.2f}")
        print(f"  创意需求: {analysis.creative_score:.2f}")
        print(f"  处理方式: {analysis.suggested_approach}")
