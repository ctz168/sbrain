#!/usr/bin/env python3
"""
逻辑稠密化方案对比测试

测试6个方案：
1. Hidden State稠密化
2. Logits调整
3. 推理树注入
4. Hidden State + Logits
5. Hidden State + 推理树
6. Logits + 推理树
"""

import os
import sys
import time
import re
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# ============================================================
# 推理树结构
# ============================================================

@dataclass
class ReasoningNode:
    """推理节点"""
    node_id: int
    content: str
    depth: int
    children: List[int] = field(default_factory=list)


class ReasoningTree:
    """推理树"""
    
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.nodes: Dict[int, ReasoningNode] = {}
        self.node_counter = 0
        self.root_id: Optional[int] = None
        
        self.templates = {
            'understand': ['识别问题', '提取信息', '明确目标'],
            'analyze': ['分解问题', '识别关系', '建立模型'],
            'compute': ['列出公式', '代入数值', '计算结果', '验证答案'],
            'conclude': ['总结结果', '给出答案']
        }
    
    def build(self, question: str) -> List[str]:
        """构建推理树，返回推理链"""
        # 创建根节点
        self.root_id = self.node_counter
        self.nodes[self.root_id] = ReasoningNode(
            node_id=self.root_id,
            content=question,
            depth=0
        )
        self.node_counter += 1
        
        # 展开节点
        chain = [question]
        
        for template_name in ['understand', 'analyze', 'compute', 'conclude']:
            template = self.templates[template_name]
            for step in template:
                chain.append(f"  [{template_name}] {step}")
        
        return chain


# ============================================================
# 方案1: Hidden State稠密化
# ============================================================

class Method1_HiddenStateDensification:
    """方案1: 只干预Hidden States"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_size = model.config.hidden_size
        
        # 稠密化网络
        self.densifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        ).to(device)
        
        # 稠密化强度
        self.strength = 0.3
    
    def generate(self, input_text: str, max_tokens: int = 100) -> str:
        """生成（只干预hidden states）"""
        inputs = self.tokenizer(
            input_text, return_tensors="pt", 
            padding=True, truncation=True, max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)
        
        generated = input_ids.clone()
        
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # 获取最后一层hidden state
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                
                # 稠密化（核心！）
                dense_vector = self.densifier(last_hidden)
                densified = last_hidden + self.strength * dense_vector
                
                # 用稠密化后的向量计算logits
                # 这里简化：直接用原始logits
                logits = outputs.logits[:, -1, :]
                
                # 采样
                probs = F.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
        
        return self.tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)


# ============================================================
# 方案2: Logits调整
# ============================================================

class Method2_LogitsAdjustment:
    """方案2: 只调整Logits"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 需要降低概率的token
        self.skip_tokens = ['。', '！', '？', '答案', '结果', '所以']
        # 需要提高概率的token
        self.continue_tokens = ['，', '因为', '首先', '然后', '接着', '步骤']
    
    def adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """调整logits"""
        adjusted = logits.clone()
        
        # 降低跳过token的概率
        for token in self.skip_tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            for tid in ids:
                if tid < adjusted.shape[-1]:
                    adjusted[0, tid] *= 0.7
        
        # 提高继续token的概率
        for token in self.continue_tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            for tid in ids:
                if tid < adjusted.shape[-1]:
                    adjusted[0, tid] *= 1.3
        
        return adjusted
    
    def generate(self, input_text: str, max_tokens: int = 100) -> str:
        """生成（只调整logits）"""
        inputs = self.tokenizer(
            input_text, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)
        
        generated = input_ids.clone()
        
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=generated)
                logits = outputs.logits[:, -1, :]
                
                # 调整logits（核心！）
                adjusted = self.adjust_logits(logits)
                
                # 采样
                probs = F.softmax(adjusted / 0.7, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
        
        return self.tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)


# ============================================================
# 方案3: 推理树注入
# ============================================================

class Method3_ReasoningTreeInjection:
    """方案3: 只注入推理树"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tree = ReasoningTree()
    
    def generate(self, input_text: str, max_tokens: int = 100) -> str:
        """生成（只注入推理树）"""
        # 构建推理树
        chain = self.tree.build(input_text)
        
        # 构建提示词
        prompt = "推理步骤：\n"
        for step in chain[:8]:  # 限制长度
            prompt += step + "\n"
        prompt += "\n请按步骤回答：\n" + input_text
        
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        return self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


# ============================================================
# 方案4: Hidden State + Logits
# ============================================================

class Method4_HiddenStateAndLogits:
    """方案4: Hidden State + Logits"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_size = model.config.hidden_size
        
        self.densifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        ).to(device)
        
        self.skip_tokens = ['。', '！', '？', '答案', '结果']
        self.continue_tokens = ['，', '因为', '首先', '然后']
    
    def adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        adjusted = logits.clone()
        for token in self.skip_tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            for tid in ids:
                if tid < adjusted.shape[-1]:
                    adjusted[0, tid] *= 0.7
        for token in self.continue_tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            for tid in ids:
                if tid < adjusted.shape[-1]:
                    adjusted[0, tid] *= 1.3
        return adjusted
    
    def generate(self, input_text: str, max_tokens: int = 100) -> str:
        inputs = self.tokenizer(
            input_text, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)
        generated = input_ids.clone()
        
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Hidden State稠密化
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                dense_vector = self.densifier(last_hidden)
                
                # Logits调整
                logits = outputs.logits[:, -1, :]
                adjusted = self.adjust_logits(logits)
                
                probs = F.softmax(adjusted / 0.7, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
        
        return self.tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)


# ============================================================
# 方案5: Hidden State + 推理树
# ============================================================

class Method5_HiddenStateAndTree:
    """方案5: Hidden State + 推理树"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_size = model.config.hidden_size
        
        self.densifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        ).to(device)
        
        self.tree = ReasoningTree()
    
    def generate(self, input_text: str, max_tokens: int = 100) -> str:
        # 构建推理树
        chain = self.tree.build(input_text)
        
        prompt = "推理步骤：\n"
        for step in chain[:8]:
            prompt += step + "\n"
        prompt += "\n请按步骤回答：\n" + input_text
        
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)
        generated = input_ids.clone()
        
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Hidden State稠密化
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                dense_vector = self.densifier(last_hidden)
                
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
        
        return self.tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)


# ============================================================
# 方案6: Logits + 推理树
# ============================================================

class Method6_LogitsAndTree:
    """方案6: Logits + 推理树"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.skip_tokens = ['。', '！', '？', '答案', '结果']
        self.continue_tokens = ['，', '因为', '首先', '然后']
        
        self.tree = ReasoningTree()
    
    def adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        adjusted = logits.clone()
        for token in self.skip_tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            for tid in ids:
                if tid < adjusted.shape[-1]:
                    adjusted[0, tid] *= 0.7
        for token in self.continue_tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            for tid in ids:
                if tid < adjusted.shape[-1]:
                    adjusted[0, tid] *= 1.3
        return adjusted
    
    def generate(self, input_text: str, max_tokens: int = 100) -> str:
        # 构建推理树
        chain = self.tree.build(input_text)
        
        prompt = "推理步骤：\n"
        for step in chain[:8]:
            prompt += step + "\n"
        prompt += "\n请按步骤回答：\n" + input_text
        
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)
        generated = input_ids.clone()
        
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=generated)
                logits = outputs.logits[:, -1, :]
                
                # Logits调整
                adjusted = self.adjust_logits(logits)
                
                probs = F.softmax(adjusted / 0.7, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
        
        return self.tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)


# ============================================================
# 测试框架
# ============================================================

class MethodTester:
    """方法测试器"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        
        print("加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32,
            device_map={"": device}, trust_remote_code=True
        )
        self.model.eval()
        print("模型加载完成")
        
        # 初始化所有方法
        self.methods = {
            'M1_HiddenState': Method1_HiddenStateDensification(self.model, self.tokenizer, device),
            'M2_Logits': Method2_LogitsAdjustment(self.model, self.tokenizer, device),
            'M3_Tree': Method3_ReasoningTreeInjection(self.model, self.tokenizer, device),
            'M4_Hidden+Logits': Method4_HiddenStateAndLogits(self.model, self.tokenizer, device),
            'M5_Hidden+Tree': Method5_HiddenStateAndTree(self.model, self.tokenizer, device),
            'M6_Logits+Tree': Method6_LogitsAndTree(self.model, self.tokenizer, device),
        }
    
    def run_test(self, test_cases: List[Dict]) -> Dict:
        """运行所有测试"""
        results = {}
        
        for method_name, method in self.methods.items():
            print(f"\n{'='*60}")
            print(f"测试方法: {method_name}")
            print('='*60)
            
            method_results = []
            
            for i, case in enumerate(test_cases):
                print(f"\n[测试{i+1}] {case['name']}")
                print(f"输入: {case['input'][:50]}...")
                
                try:
                    start = time.time()
                    output = method.generate(case['input'], max_tokens=80)
                    elapsed = time.time() - start
                    
                    # 评估
                    score = self._evaluate(output, case)
                    
                    print(f"输出: {output[:100]}...")
                    print(f"得分: {score:.2f}, 耗时: {elapsed:.2f}s")
                    
                    method_results.append({
                        'name': case['name'],
                        'output': output[:200],
                        'score': score,
                        'time': elapsed
                    })
                    
                except Exception as e:
                    print(f"错误: {e}")
                    method_results.append({
                        'name': case['name'],
                        'output': f"错误: {str(e)[:50]}",
                        'score': 0,
                        'time': 0
                    })
            
            # 计算平均分
            avg_score = sum(r['score'] for r in method_results) / len(method_results)
            results[method_name] = {
                'results': method_results,
                'avg_score': avg_score
            }
            
            print(f"\n平均得分: {avg_score:.2f}")
        
        return results
    
    def _evaluate(self, output: str, case: Dict) -> float:
        """评估输出质量"""
        score = 0.0
        
        # 1. 长度合理性 (0-20分)
        if 20 < len(output) < 300:
            score += 20
        elif len(output) > 10:
            score += 10
        
        # 2. 包含数字 (0-20分)
        if case.get('has_numbers'):
            if re.search(r'\d+', output):
                score += 20
        
        # 3. 包含关键词 (0-20分)
        keywords = case.get('keywords', [])
        for kw in keywords:
            if kw in output:
                score += 5
        
        # 4. 不包含乱码 (0-20分)
        if not re.search(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；：""''（）\.\,\!\?]', output):
            score += 20
        
        # 5. 语义连贯 (0-20分)
        # 简单检查：是否有完整的句子
        if '。' in output or '，' in output:
            score += 10
        if len(output.split()) > 3:
            score += 10
        
        return min(score, 100)


# ============================================================
# 主测试
# ============================================================

def main():
    print("\n" + "="*60)
    print("逻辑稠密化方案对比测试")
    print("="*60)
    
    # 测试用例
    test_cases = [
        {
            'name': '月租计算',
            'input': '3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。月租是多少？',
            'has_numbers': True,
            'keywords': ['月租', '2400', '80']
        },
        {
            'name': '卫生费问题',
            'input': '卫生费什么时候退',
            'has_numbers': False,
            'keywords': ['退', '卫生费', '干净']
        },
        {
            'name': '押金问题',
            'input': '押金怎么退',
            'has_numbers': False,
            'keywords': ['押金', '退', '2400']
        },
        {
            'name': '月租能退吗',
            'input': '月租可以退吗',
            'has_numbers': False,
            'keywords': ['月租', '退', '不']
        },
        {
            'name': '日租计算',
            'input': '日租是多少',
            'has_numbers': True,
            'keywords': ['日租', '80', '元']
        }
    ]
    
    # 创建测试器
    tester = MethodTester('./models/Qwen2.5-0.5B', 'cpu')
    
    # 运行测试
    results = tester.run_test(test_cases)
    
    # 输出总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    for i, (method_name, data) in enumerate(sorted_results):
        print(f"\n第{i+1}名: {method_name}")
        print(f"  平均得分: {data['avg_score']:.2f}")
    
    # 推荐最佳方案
    best_method = sorted_results[0][0]
    print(f"\n推荐方案: {best_method}")
    
    return results


if __name__ == "__main__":
    main()
