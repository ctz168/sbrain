#!/usr/bin/env python3
"""
Telegram Bot - O1连续计算版

核心特性：
- 动态聚焦窄窗口注意力: O(n²) → O(1)
- 持续思维状态: 不重置
- 记忆锚点系统
- 连续推理链
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


class O1ContinuousBot:
    """
    O1连续计算Bot
    
    核心创新：
    - 动态聚焦窄窗口注意力: O(n²) → O(1)
    - 持续思维状态: 不重置
    - 记忆锚点系统
    """
    
    def __init__(self, token: str, model_path: str, device: str = "cpu"):
        self.token = token
        self.device = device
        
        print("\n" + "=" * 60)
        print("O1连续计算类脑AI - Telegram Bot")
        print("=" * 60)
        print(f"模型路径: {model_path}")
        print(f"设备: {device}")
        print("=" * 60)
        
        # 加载O1连续模型
        print("\n正在初始化O1连续计算类脑AI...")
        from core.o1_continuous_brain import O1ContinuousBrain
        self.brain = O1ContinuousBrain(model_path=model_path, device=device)
        print("\n✓ Bot 就绪")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /start 命令"""
        # 重置思维状态
        self.brain.reset()
        
        await update.message.reply_text(
            "🧠 O1连续计算类脑AI\n\n"
            "核心创新：\n"
            "• 动态聚焦窄窗口注意力\n"
            "  - O(n²) → O(1)\n"
            "  - 算力开销完全固定\n\n"
            "• 持续思维状态\n"
            "  - 不重置\n"
            "  - 连续推理链\n\n"
            "• 记忆锚点系统\n"
            "  - 自动创建\n"
            "  - 动态检索\n\n"
            "思维状态已重置，请开始对话："
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /stats 命令"""
        stats = self.brain.get_stats()
        
        msg = "📊 系统统计\n\n"
        msg += "系统状态:\n"
        msg += f"  总查询数: {stats['system']['total_queries']}\n"
        msg += f"  总Token数: {stats['system']['total_tokens']}\n"
        msg += f"  总锚点数: {stats['system']['total_anchors']}\n"
        msg += f"  平均检索锚点: {stats['system']['avg_anchors_retrieved']:.2f}\n\n"
        msg += "思维引擎:\n"
        msg += f"  推理链长度: {stats['thought_engine']['reasoning_chain_length']}\n"
        msg += f"  缓存关键信息: {stats['thought_engine']['key_info_cached']}\n"
        
        await update.message.reply_text(msg)
    
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /reset 命令"""
        self.brain.reset()
        await update.message.reply_text("思维状态已重置，可以开始新的对话。")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理消息"""
        user_input = update.message.text
        user_id = update.effective_user.id
        
        logger.info(f"收到用户 {user_id} 消息：{user_input[:50]}...")
        
        # 发送"思考中"消息
        thinking_msg = await update.message.reply_text("🤔 O1连续思维中...")
        
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
        application.add_handler(CommandHandler("reset", self.reset_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # 启动
        logger.info("正在启动 Telegram Bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    parser = argparse.ArgumentParser(description="O1连续计算类脑AI Telegram Bot")
    parser.add_argument("--token", type=str, required=True, help="Telegram Bot Token")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    
    args = parser.parse_args()
    
    model_path = "./models/DeepSeek-R1-Distill-Qwen-1.5B"
    
    bot = O1ContinuousBot(
        token=args.token,
        model_path=model_path,
        device=args.device
    )
    
    bot.run()


if __name__ == "__main__":
    main()
