#!/usr/bin/env python3
"""
Telegram Bot - 真正的O1连续计算版

使用真正的O1连续计算实现：
1. 记忆锚点真正用于生成
2. 关键信息真正用于生成
3. 持续思维状态真正影响生成
"""

import os
import sys
import asyncio
import logging
from typing import Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import argparse

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TrueO1Bot:
    """真正的O1连续计算Bot"""
    
    def __init__(self, token: str, model_path: str, device: str = "cpu"):
        self.token = token
        self.device = device
        
        print("\n" + "=" * 60)
        print("真正的O1连续计算类脑AI - Telegram Bot")
        print("=" * 60)
        
        from core.true_o1_brain import TrueO1ContinuousBrain
        self.brain = TrueO1ContinuousBrain(model_path=model_path, device=device)
        print("\n✓ Bot 就绪")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.brain.reset()
        await update.message.reply_text(
            "🧠 真正的O1连续计算类脑AI\n\n"
            "核心特性：\n"
            "• 记忆锚点真正用于生成\n"
            "• 关键信息真正用于生成\n"
            "• 持续思维状态真正影响生成\n"
            "• 真正的连续计算\n\n"
            "思维状态已重置，请开始对话："
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = self.brain.get_stats()
        
        msg = "📊 系统统计\n\n"
        msg += f"总查询数: {stats['system']['total_queries']}\n"
        msg += f"总Token数: {stats['system']['total_tokens']}\n"
        msg += f"创建锚点: {stats['system']['anchors_created']}\n"
        msg += f"使用锚点: {stats['system']['anchors_used']}\n\n"
        msg += "思维引擎:\n"
        msg += f"  对话轮数: {stats['thought_engine']['dialogue_turns']}\n"
        msg += f"  缓存数字: {stats['thought_engine']['cached_numbers']}\n"
        msg += f"  缓存实体: {stats['thought_engine']['cached_entities']}\n"
        msg += f"  缓存关系: {stats['thought_engine']['cached_relations']}\n"
        
        await update.message.reply_text(msg)
    
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.brain.reset()
        await update.message.reply_text("思维状态已重置。")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_input = update.message.text
        user_id = update.effective_user.id
        
        logger.info(f"收到用户 {user_id} 消息：{user_input[:50]}...")
        
        thinking_msg = await update.message.reply_text("🤔 O1连续思维中...")
        
        try:
            response_text = ""
            
            async for char in self.brain.generate_stream(user_input):
                response_text += char
            
            logger.info(f"生成完成，长度: {len(response_text)}")
            
            if response_text and response_text.strip():
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
    
    def run(self):
        print("\n启动服务... (Ctrl+C 停止)")
        
        application = Application.builder().token(self.token).build()
        
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("reset", self.reset_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        logger.info("正在启动 Telegram Bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    parser = argparse.ArgumentParser(description="真正的O1连续计算类脑AI Telegram Bot")
    parser.add_argument("--token", type=str, required=True, help="Telegram Bot Token")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    
    args = parser.parse_args()
    
    model_path = "./models/DeepSeek-R1-Distill-Qwen-1.5B"
    
    bot = TrueO1Bot(
        token=args.token,
        model_path=model_path,
        device=args.device
    )
    
    bot.run()


if __name__ == "__main__":
    main()
