#!/usr/bin/env python3
"""
Telegram Bot - 连续密度场版

使用连续逻辑密度场，逐token实时处理
"""

import os
import sys
import asyncio
import logging
from typing import Optional

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import argparse

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class ContinuousDensityBot:
    """
    连续密度场Bot
    
    核心特点：
    - 逐token实时处理
    - 连续密度场动态调整
    - 类人脑处理方式
    """
    
    def __init__(self, token: str, model_path: str, device: str = "cpu"):
        self.token = token
        self.device = device
        
        print("\n" + "=" * 60)
        print("连续密度场类脑AI - Telegram Bot")
        print("=" * 60)
        print(f"模型路径: {model_path}")
        print(f"设备: {device}")
        print("=" * 60)
        
        # 加载模型
        print("\n正在初始化连续密度场类脑AI...")
        from core.continuous_brain import ContinuousDensityBrain
        self.brain = ContinuousDensityBrain(model_path=model_path, device=device)
        print("\n✓ Bot 就绪")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /start 命令"""
        await update.message.reply_text(
            "🧠 连续密度场类脑AI\n\n"
            "核心特点：\n"
            "- 逐token实时处理\n"
            "- 连续逻辑密度场\n"
            "- 动态调整推理强度\n\n"
            "请输入您的问题："
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /stats 命令"""
        stats = self.brain.get_stats()
        
        msg = "📊 系统统计\n\n"
        msg += f"总查询数: {stats['system']['total_queries']}\n"
        msg += f"总Token数: {stats['system']['total_tokens']}\n"
        msg += f"平均密度: {stats['system']['avg_density']:.2f}\n\n"
        msg += "密度分布:\n"
        msg += f"  高密度: {stats['density_distribution']['high']*100:.1f}%\n"
        msg += f"  中密度: {stats['density_distribution']['medium']*100:.1f}%\n"
        msg += f"  低密度: {stats['density_distribution']['low']*100:.1f}%\n"
        
        await update.message.reply_text(msg)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理消息"""
        user_input = update.message.text
        user_id = update.effective_user.id
        
        logger.info(f"收到用户 {user_id} 消息：{user_input[:50]}...")
        
        # 发送"思考中"消息
        thinking_msg = await update.message.reply_text("🤔 思考中...")
        
        try:
            # 流式生成
            response_text = ""
            
            async for char in self.brain.generate_stream(user_input):
                response_text += char
            
            logger.info(f"生成完成，长度: {len(response_text)}")
            
            # 最终更新
            if response_text and response_text.strip():
                # 清理输出
                response_text = self._clean_output(response_text)
                
                try:
                    await thinking_msg.edit_text(
                        response_text[:4000] if len(response_text) > 4000 else response_text
                    )
                except Exception as e:
                    logger.warning(f"编辑消息失败: {e}")
                    await update.message.reply_text(response_text[:4000])
                
                logger.info(f"回复用户 {user_id}: {response_text[:50]}...")
            else:
                logger.warning("生成内容为空")
                await thinking_msg.edit_text("生成内容为空，请重试。")
                
        except Exception as e:
            logger.error(f"生成失败: {e}")
            import traceback
            traceback.print_exc()
            await thinking_msg.edit_text(f"生成出错: {str(e)[:100]}")
    
    def _clean_output(self, text: str) -> str:
        """清理输出"""
        if not text or not text.strip():
            return "生成内容为空"
        
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
                if cleaned and cleaned[-1].strip():
                    cleaned.append(line)
        
        result = '\n'.join(cleaned)
        
        # 限制长度
        if len(result) > 800:
            last_period = result.rfind('。', 0, 800)
            if last_period > 200:
                result = result[:last_period + 1]
            else:
                result = result[:800] + "..."
        
        return result if result.strip() else text[:500]
    
    def run(self):
        """运行Bot"""
        print("\n启动服务... (Ctrl+C 停止)")
        print("=" * 60)
        
        # 创建应用
        application = Application.builder().token(self.token).build()
        
        # 添加处理器
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # 启动
        logger.info("正在启动 Telegram Bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    parser = argparse.ArgumentParser(description="连续密度场类脑AI Telegram Bot")
    parser.add_argument("--token", type=str, required=True, help="Telegram Bot Token")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--quantization", type=str, default="FP32", help="量化方式")
    
    args = parser.parse_args()
    
    model_path = "./models/DeepSeek-R1-Distill-Qwen-1.5B"
    
    bot = ContinuousDensityBot(
        token=args.token,
        model_path=model_path,
        device=args.device
    )
    
    bot.run()


if __name__ == "__main__":
    main()
