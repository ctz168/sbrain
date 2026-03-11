#!/usr/bin/env python3
"""
LSDC 引擎 - Telegram Bot

集成逻辑自相似稠密补齐引擎的Telegram Bot
"""

import os
import sys
import asyncio
import logging
import argparse
from typing import Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from lsdc_engine.logic_processor import create_logic_processor, LogicProcessor

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class LSDCBot:
    """LSDC Telegram Bot"""
    
    def __init__(self, token: str, model_path: str, device: str = "cpu"):
        self.token = token
        self.device = device
        
        print("\n" + "=" * 60)
        print("LSDC 引擎 - Telegram Bot")
        print("=" * 60)
        
        # 初始化逻辑处理器
        print("\n初始化逻辑处理器...")
        self.processor = create_logic_processor(model_path, device)
        
        # 会话状态
        self.sessions: dict = {}
        
        print("\n" + "=" * 60)
        print("✓ Bot 就绪")
        print("=" * 60)
    
    def get_session(self, user_id: int) -> dict:
        """获取用户会话"""
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                'context': None,
                'turn_count': 0
            }
        return self.sessions[user_id]
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """开始命令"""
        user_id = update.effective_user.id
        self.sessions[user_id] = {
            'context': None,
            'turn_count': 0
        }
        
        await update.message.reply_text(
            "🧠 LSDC 引擎\n\n"
            "逻辑自相似稠密补齐引擎\n\n"
            "核心特性：\n"
            "• 离散状态转移\n"
            "• 窄宽带补齐\n"
            "• 自相似结构\n\n"
            "请开始对话："
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """统计命令"""
        user_id = update.effective_user.id
        session = self.get_session(user_id)
        
        chain = self.processor.get_chain()
        chain_length = len(chain.nodes) if chain else 0
        
        msg = f"📊 会话统计\n\n"
        msg += f"对话轮数: {session['turn_count']}\n"
        msg += f"逻辑节点: {chain_length}\n"
        
        await update.message.reply_text(msg)
    
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """重置命令"""
        user_id = update.effective_user.id
        self.sessions[user_id] = {
            'context': None,
            'turn_count': 0
        }
        await update.message.reply_text("会话已重置。")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理消息"""
        user_input = update.message.text
        user_id = update.effective_user.id
        
        logger.info(f"用户 {user_id}: {user_input[:40]}...")
        
        # 获取会话
        session = self.get_session(user_id)
        session['turn_count'] += 1
        
        thinking_msg = await update.message.reply_text("🧠 LSDC处理中...")
        
        try:
            # 使用LSDC引擎处理
            response_text = ""
            node_count = 0
            
            for node in self.processor.process(user_input, session['context']):
                node_count += 1
                
                # 构建响应
                if node.premise:
                    response_text += f"【前提】{node.premise[:50]}\n"
                if node.derivation:
                    response_text += f"【推演】{node.derivation[:80]}\n"
                if node.conclusion:
                    response_text += f"【结论】{node.conclusion[:50]}\n"
                response_text += f"[密度: {node.density:.1f}]\n\n"
                
                # 更新上下文
                session['context'] = node.conclusion
            
            # 添加统计
            response_text += f"---\n节点数: {node_count} | 轮数: {session['turn_count']}"
            
            logger.info(f"生成完成，节点数: {node_count}")
            
            # 发送响应
            if response_text:
                await thinking_msg.edit_text(response_text[:4000])
            else:
                await thinking_msg.edit_text("生成内容为空，请重试。")
                
        except Exception as e:
            logger.error(f"生成失败: {e}")
            import traceback
            traceback.print_exc()
            await thinking_msg.edit_text(f"生成出错: {str(e)[:100]}")
    
    def run(self):
        """运行Bot"""
        print("\n启动服务... (Ctrl+C 停止)")
        
        application = Application.builder().token(self.token).build()
        
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("reset", self.reset_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        logger.info("启动 Telegram Bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    parser = argparse.ArgumentParser(description="LSDC Telegram Bot")
    parser.add_argument("--token", type=str, required=True, help="Telegram Bot Token")
    parser.add_argument("--model", type=str, default="../models/Qwen3.5-0.8B")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    bot = LSDCBot(
        token=args.token,
        model_path=args.model,
        device=args.device
    )
    
    bot.run()


if __name__ == "__main__":
    main()
