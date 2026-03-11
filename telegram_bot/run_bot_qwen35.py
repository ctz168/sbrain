#!/usr/bin/env python3
"""
Telegram Bot - Qwen3.5-0.8B版本

使用Qwen3.5-0.8B模型，经过充分测试
"""

import os
import sys
import asyncio
import logging
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram Bot"""
    
    def __init__(self, token: str, model_path: str, device: str = "cpu"):
        self.token = token
        self.device = device
        
        print("\n" + "=" * 60)
        print("Qwen3.5-0.8B Telegram Bot")
        print("=" * 60)
        
        from core.qwen35_bot import Qwen35ContinuousBot
        self.bot = Qwen35ContinuousBot(model_path=model_path, device=device)
        print("\n✓ Bot 就绪")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.bot.reset()
        await update.message.reply_text(
            "🧠 Qwen3.5-0.8B 连续对话Bot\n\n"
            "特性：\n"
            "• 推理树注入\n"
            "• 上下文记忆\n"
            "• 重复检测\n"
            "• 计算问题特殊处理\n\n"
            "请开始对话："
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = self.bot.get_stats()
        msg = f"📊 统计\n\n"
        msg += f"总查询数: {stats['total_queries']}\n"
        msg += f"总Token数: {stats['total_tokens']}\n"
        msg += f"对话轮数: {stats['dialogue_turns']}\n"
        await update.message.reply_text(msg)
    
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.bot.reset()
        await update.message.reply_text("状态已重置。")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_input = update.message.text
        user_id = update.effective_user.id
        
        logger.info(f"用户 {user_id}: {user_input[:40]}...")
        
        thinking_msg = await update.message.reply_text("🤔 思考中...")
        
        try:
            response_text = ""
            async for char in self.bot.generate_stream(user_input):
                response_text += char
            
            logger.info(f"生成完成，长度: {len(response_text)}")
            
            if response_text and response_text.strip():
                try:
                    await thinking_msg.edit_text(
                        response_text[:4000] if len(response_text) > 4000 else response_text
                    )
                except Exception as e:
                    logger.warning(f"编辑失败: {e}")
                    await update.message.reply_text(response_text[:4000])
            else:
                await thinking_msg.edit_text("生成内容为空，请重试。")
                
        except Exception as e:
            logger.error(f"生成失败: {e}")
            import traceback
            traceback.print_exc()
            await thinking_msg.edit_text(f"生成出错: {str(e)[:100]}")
    
    def run(self):
        print("\n启动服务... (Ctrl+C 停止)")
        
        application = Application.builder().token(self.token).build()
        
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("reset", self.reset_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        logger.info("启动 Telegram Bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B Telegram Bot")
    parser.add_argument("--token", type=str, required=True, help="Telegram Bot Token")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    
    args = parser.parse_args()
    
    model_path = "./models/Qwen3.5-0.8B"
    
    bot = TelegramBot(
        token=args.token,
        model_path=model_path,
        device=args.device
    )
    
    bot.run()


if __name__ == "__main__":
    main()
