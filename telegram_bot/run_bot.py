#!/usr/bin/env python3
"""
类人脑双系统AI - Telegram Bot 启动脚本 (优化版v2)
"""

import os
import sys
import argparse
import re

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(description='类人脑双系统AI Telegram Bot')
    parser.add_argument('--token', type=str, required=True, help='Telegram Bot Token')
    parser.add_argument('--model-path', type=str, default='./models/Qwen2.5-0.5B', help='模型路径')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='运行设备')
    parser.add_argument('--quantization', type=str, default='FP32', choices=['FP32', 'INT4', 'INT8'], help='量化类型')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("类人脑双系统全闭环AI架构 - Telegram Bot")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"量化: {args.quantization}")
    print("=" * 60)
    
    # 检查模型路径
    model_path = args.model_path
    if not os.path.exists(model_path):
        alt_path = os.path.join(project_root, "models/Qwen2.5-0.5B")
        if os.path.exists(alt_path):
            model_path = alt_path
            print(f"使用模型: {model_path}")
        else:
            print(f"\n❌ 模型未找到: {model_path}")
            sys.exit(1)
    
    # 加载模型
    print("\n正在加载模型...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        print(f"✓ Tokenizer 加载成功，词表大小：{len(tokenizer)}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": args.device},
            trust_remote_code=True
        )
        model.eval()
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型加载成功，参数量：{param_count:,} ({param_count/1e6:.2f}M)")
        
        # 创建AI接口
        ai_interface = SimpleAIInterface(model, tokenizer, args.device)
        
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 启动Bot
    print("\n正在启动Bot服务...")
    try:
        from telegram_bot.bot import BrainAIBot
        
        bot = BrainAIBot(token=args.token, ai_interface=ai_interface)
        print("✓ Bot 就绪")
        print("\n启动服务... (Ctrl+C 停止)")
        print("=" * 60)
        bot.run()
        
    except Exception as e:
        print(f"❌ Bot启动失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


class SimpleAIInterface:
    """简化的AI接口 (优化版v2)"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.total_tokens = 0
        self.total_time = 0.0
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def _detect_question_type(self, text: str) -> str:
        """检测问题类型"""
        if any(kw in text for kw in ['月租', '房租', '计算', '多少', '合计', '押金', '卫生费']):
            return 'calculation'
        elif any(kw in text for kw in ['你好', '您好', 'hello', 'hi']):
            return 'greeting'
        elif any(kw in text for kw in ['什么', '为什么', '怎么', '如何']):
            return 'qa'
        else:
            return 'general'
    
    def _build_prompt(self, input_text: str, question_type: str) -> str:
        """构建优化的提示词"""
        if question_type == 'calculation':
            # 提取数字信息
            numbers = re.findall(r'[\d,]+\.?\d*', input_text.replace('，', ',').replace('。', '.'))
            
            prompt = f"""请根据以下信息计算月租：

{input_text}

计算步骤：
1. 首先识别各项费用
2. 押金是可退还的，不计入月租
3. 卫生费如果可退还，也不计入月租
4. 计算实际月租金

请直接给出答案："""
        
        elif question_type == 'greeting':
            prompt = f"用户说：{input_text}\n请友好地回复："
        
        else:
            prompt = f"用户问：{input_text}\n请简洁回答："
        
        return prompt
    
    async def generate_stream(self, input_text: str, max_tokens: int = 200):
        """流式生成 (优化版v2)"""
        import torch
        import asyncio
        
        # 检测问题类型并构建提示词
        question_type = self._detect_question_type(input_text)
        prompt = self._build_prompt(input_text, question_type)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # 生成参数根据问题类型调整
        if question_type == 'calculation':
            temperature = 0.1  # 计算类问题用更低温度
            top_p = 0.8
        else:
            temperature = 0.5
            top_p = 0.9
        
        # 生成
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                top_k=30,
                repetition_penalty=1.5,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # 解码
        generated_text = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 清理和格式化
        generated_text = self._clean_output(generated_text, question_type)
        
        # 流式输出
        for char in generated_text:
            yield char
            await asyncio.sleep(0.02)
    
    def _clean_output(self, text: str, question_type: str) -> str:
        """清理输出"""
        # 移除重复行
        lines = text.split('\n')
        seen = set()
        cleaned = []
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                cleaned.append(line)
            elif not line_stripped:
                if cleaned and cleaned[-1].strip():  # 保留单个空行
                    cleaned.append(line)
        
        result = '\n'.join(cleaned)
        
        # 限制长度
        if len(result) > 400:
            # 尝试在句号处截断
            last_period = result.rfind('。', 0, 400)
            if last_period > 200:
                result = result[:last_period + 1]
            else:
                result = result[:400] + "..."
        
        return result
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'system': {
                'total_tokens': self.total_tokens,
                'total_time': self.total_time,
                'avg_time_per_token': 0,
                'device': self.device,
                'quantization': 'FP32'
            }
        }


if __name__ == "__main__":
    main()
