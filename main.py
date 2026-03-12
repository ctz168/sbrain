#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 主入口

功能：
- 模型加载与初始化
- 交互式对话
- 模型训练
- 测评运行
"""

import os
import sys
import argparse
import torch

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from configs.config import BrainConfig, default_config
from core.model import BrainAIModel, create_model
from evaluation.evaluator import EvaluationSystem


def main():
    parser = argparse.ArgumentParser(description='类人脑双系统全闭环AI架构')
    parser.add_argument('--mode', type=str, default='chat',
                       choices=['chat', 'train', 'eval', 'serve'],
                       help='运行模式')
    parser.add_argument('--model-path', type=str, default=None,
                       help='模型路径')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='运行设备')
    parser.add_argument('--quantization', type=str, default='FP32',
                       choices=['FP32', 'INT4', 'INT8'],
                       help='量化类型')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("类人脑双系统全闭环AI架构")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"设备: {args.device}")
    print(f"量化: {args.quantization}")
    print("=" * 60)
    
    # 加载配置
    config = default_config
    config.device = args.device
    config.quantization = args.quantization
    
    # 创建模型
    print("\n正在加载模型...")
    model = create_model(
        model_path=args.model_path or config.model_path,
        config=config
    )
    
    if args.mode == 'chat':
        run_chat(model)
    elif args.mode == 'train':
        run_train(model)
    elif args.mode == 'eval':
        run_eval(model)
    elif args.mode == 'serve':
        run_serve(model)


def run_chat(model: BrainAIModel):
    """交互式对话模式"""
    print("\n" + "=" * 60)
    print("交互式对话模式")
    print("输入 'quit' 退出, 'stats' 查看统计, 'save' 保存检查点")
    print("=" * 60 + "\n")
    
    history = []
    
    while True:
        try:
            user_input = input("用户: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\n再见!")
                break
            
            if user_input.lower() == 'stats':
                stats = model.get_stats()
                print("\n系统统计:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
                continue
            
            if user_input.lower() == 'save':
                model.save_checkpoint('checkpoint.pt')
                print("检查点已保存\n")
                continue
            
            # 生成回复
            print("助手: ", end="", flush=True)
            response = model.chat(user_input, history)
            print(response)
            print()
            
            # 更新历史
            history.append({'role': 'user', 'content': user_input})
            history.append({'role': 'assistant', 'content': response})
            
            # 保持历史长度
            if len(history) > 10:
                history = history[-10:]
            
            # 尝试记忆巩固
            consolidation = model.consolidate_memory()
            if consolidation:
                print(f"[记忆巩固] 回放 {consolidation.get('replayed_memories', 0)} 条记忆")
        
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")


def run_train(model: BrainAIModel):
    """训练模式"""
    print("\n" + "=" * 60)
    print("训练模式")
    print("=" * 60)
    
    # 场景预适配训练
    print("\n执行场景预适配训练...")
    
    # 模拟训练数据
    train_data = {
        'general_dialog': [{'input': '你好', 'output': '你好！有什么可以帮助你的？'}] * 10,
        'logical_reasoning': [{'input': '请推理', 'output': '好的，让我分析一下...'}] * 10,
        'code_generation': [{'input': '写代码', 'output': '好的，我来写...'}] * 10,
    }
    
    results = model.scene_adapt.pretrain_all_scenes(model, train_data)
    
    print("\n训练结果:")
    for scene, result in results.items():
        print(f"  {scene}: loss={result.get('final_loss', 0):.4f}")
    
    # 保存检查点
    model.save_checkpoint('trained_model.pt')
    print("\n模型已保存")


def run_eval(model: BrainAIModel):
    """测评模式"""
    print("\n" + "=" * 60)
    print("测评模式")
    print("=" * 60)
    
    # 创建测评系统
    eval_system = EvaluationSystem(model.config.evaluation)
    
    # 运行完整测评
    print("\n正在执行测评...")
    report = eval_system.run_full_evaluation(
        model=model,
        tokenizer=model.tokenizer,
        hippocampus_system=model.hippocampus,
        metacognition_system=model.metacognition,
        stdp_controller=model.stdp_controller
    )
    
    # 生成报告
    report_text = eval_system.generate_report(report, 'evaluation_report.txt')
    print(report_text)


def run_serve(model: BrainAIModel):
    """服务模式"""
    print("\n" + "=" * 60)
    print("服务模式")
    print("=" * 60)
    
    # 简单的HTTP服务
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        class ModelHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                message = data.get('message', '')
                response = model.chat(message)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                response_data = {'response': response}
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            
            def log_message(self, format, *args):
                print(f"[HTTP] {args[0]}")
        
        server = HTTPServer(('0.0.0.0', 8080), ModelHandler)
        print("服务已启动: http://0.0.0.0:8080")
        print("POST / 请求体: {\"message\": \"你的消息\"}")
        server.serve_forever()
    
    except ImportError:
        print("HTTP服务不可用，请安装所需依赖")
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == '__main__':
    main()
