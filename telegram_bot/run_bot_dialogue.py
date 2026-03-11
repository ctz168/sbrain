#!/usr/bin/env python3
"""
Telegram Bot - 连续对话版

解决"断片"问题：
- 对话历史缓冲区
- 上下文注入
- 连续对话流
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


class ContinuousDialogueBot:
    """
    连续对话Bot
    
    解决"断片"问题：
    - 对话历史缓冲区
    - 上下文注入
    - 连续对话流
    """
    
    def __init__(self, token: str, model_path: str, device: str = "cpu"):
        self.token = token
        self.device = device
        
        print("\n" + "=" * 60)
        print("连续对话类脑AI - Telegram Bot")
        print("=" * 60)
        print(f"模型路径: {model_path}")
        print(f"设备: {device}")
        print("=" * 60)
        
        # 加载连续对话模型
        print("\n正在初始化连续对话类脑AI...")
        from core.continuous_dialogue_brain import ContinuousDialogueBrain
        self.brain = ContinuousDialogueBrain(model_path=model_path, device=device)
        print("\n✓ Bot 就绪")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /start 命令"""
        # 清空对话历史
        self.brain.clear_history()
        
        await update.message.reply_text(
            "🧠 连续对话类脑AI\n\n"
            "核心特性：\n"
            "• 对话历史记忆（不断片）\n"
            "• 上下文连续性\n"
            "• 连续逻辑密度场\n"
            "• 自相似逻辑链稠密化\n\n"
            "对话历史已清空，请开始新的对话："
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /stats 命令"""
        stats = self.brain.get_stats()
        
        msg = "📊 系统统计\n\n"
        msg += f"总查询数: {stats['system']['total_queries']}\n"
        msg += f"总Token数: {stats['system']['total_tokens']}\n"
        msg += f"平均密度: {stats['system']['avg_density']:.2f}\n"
        msg += f"对话轮数: {stats['system']['dialogue_turns']}\n\n"
        msg += "模块状态:\n"
        msg += f"  STDP更新: {stats['modules']['stdp_updates']}\n"
        msg += f"  记忆存储: {stats['modules']['memory_stores']}\n"
        
        await update.message.reply_text(msg)
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /clear 命令"""
        self.brain.clear_history()
        await update.message.reply_text("对话历史已清空，可以开始新的对话。")
    
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
        """运行Bot"""
        print("\n启动服务... (Ctrl+C 停止)")
        print("=" * 60)
        
        # 创建应用
        application = Application.builder().token(self.token).build()
        
        # 添加处理器
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("clear", self.clear_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # 启动
        logger.info("正在启动 Telegram Bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    parser = argparse.ArgumentParser(description="连续对话类脑AI Telegram Bot")
    parser.add_argument("--token", type=str, required=True, help="Telegram Bot Token")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    
    args = parser.parse_args()
    
    model_path = "./models/Qwen2.5-0.5B"
    
    bot = ContinuousDialogueBot(
        token=args.token,
        model_path=model_path,
        device=args.device
    )
    
    bot.run()


if __name__ == "__main__":
    main()
