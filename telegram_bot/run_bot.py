#!/usr/bin/env python3
"""
类人脑双系统AI - Telegram Bot 启动脚本
"""

import os
import sys
import argparse

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
            print(f"请运行: huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ./models/Qwen2.5-0.5B")
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
    """简化的AI接口"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.total_tokens = 0
        self.total_time = 0.0
    
    async def generate_stream(self, input_text: str, max_tokens: int = 200):
        """流式生成"""
        import torch
        import asyncio
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # 生成
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        generated_text = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 模拟流式输出
        for char in generated_text:
            yield char
            await asyncio.sleep(0.02)
    
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
