"""
Telegram Bot 主程序

实现与类人脑AI模型的交互，支持流式输出
"""

import asyncio
import logging
import time
from typing import Optional, Dict, List
from telegram import Update, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class BrainAIBot:
    """
    类人脑AI Telegram Bot
    
    功能:
    - 与用户进行自然语言对话
    - 支持流式输出
    - 显示打字状态
    - 多轮对话上下文管理
    """
    
    def __init__(
        self,
        token: str,
        ai_interface=None,
        stream_chunk_size: int = 1,
        stream_delay_ms: int = 50,
        max_context_length: int = 10
    ):
        """
        初始化 Bot
        
        Args:
            token: Telegram Bot Token
            ai_interface: BrainAIInterface 实例
            stream_chunk_size: 流式输出块大小
            stream_delay_ms: 流式延迟 (毫秒)
            max_context_length: 最大上下文长度
        """
        self.token = token
        self.ai = ai_interface
        self.stream_chunk_size = stream_chunk_size
        self.stream_delay_ms = stream_delay_ms
        self.max_context_length = max_context_length
        
        # 用户对话历史
        self.user_history: Dict[int, List[Dict[str, str]]] = {}
        
        # Bot 应用
        self.application: Optional[Application] = None
        
        # 打字模拟器
        self.typing_tasks: Dict[int, asyncio.Task] = {}
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /start 命令"""
        user_id = update.effective_user.id
        
        welcome_message = (
            "🧠 欢迎使用类人脑双系统AI助手！\n\n"
            "✨ 特性:\n"
            "• 基于DeepSeek-R1-Distill-Qwen-1.5B模型\n"
            "• 海马体-新皮层双系统架构\n"
            "• 10ms/100Hz高刷新推理\n"
            "• STDP在线学习\n"
            "• 元认知双闭环校验\n\n"
            "💡 直接发送消息即可与我对话！\n"
            "使用 /help 查看更多帮助"
        )
        
        await update.message.reply_text(welcome_message)
        logger.info(f"User {user_id} started the bot")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /help 命令"""
        help_text = (
            "📚 帮助信息\n\n"
            "可用命令:\n"
            "/start - 重新开始对话\n"
            "/help - 显示帮助信息\n"
            "/clear - 清除对话历史\n"
            "/stats - 查看系统统计\n\n"
            "💬 直接发送消息即可与我对话！\n"
            "我会实时显示思考过程 (流式输出)"
        )
        
        await update.message.reply_text(help_text)
        logger.info(f"User {update.effective_user.id} requested help")
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /clear 命令"""
        user_id = update.effective_user.id
        
        if user_id in self.user_history:
            del self.user_history[user_id]
        
        await update.message.reply_text("✓ 对话历史已清除")
        logger.info(f"Cleared history for user {user_id}")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /stats 命令"""
        if not self.ai:
            await update.message.reply_text("⚠️ AI 模型未初始化")
            return
        
        try:
            stats = self.ai.get_stats()
            
            stats_text = (
                "📊 系统统计\n\n"
                f"🧠 总Token数：{stats.get('system', {}).get('total_tokens', 0)}\n"
                f"⏱️ 总时间：{stats.get('system', {}).get('total_time', 0):.2f}s\n"
                f"⚡ 平均延迟：{stats.get('system', {}).get('avg_time_per_token', 0):.2f}ms\n"
                f"🖥️ 设备：{stats.get('system', {}).get('device', 'cpu')}\n"
                f"📦 量化：{stats.get('system', {}).get('quantization', 'FP32')}"
            )
            
            await update.message.reply_text(stats_text)
        except Exception as e:
            await update.message.reply_text(f"❌ 获取统计失败：{e}")
    
    async def _typing_loop(self, bot, chat_id):
        """打字状态循环"""
        while True:
            try:
                await bot.send_chat_action(chat_id=chat_id, action="typing")
                await asyncio.sleep(4)
            except asyncio.CancelledError:
                break
            except Exception:
                break
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理用户消息"""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        user_message = update.message.text
        
        if not user_message:
            return
        
        logger.info(f"收到用户 {user_id} 消息：{user_message[:50]}...")
        
        # 开始打字状态
        if chat_id in self.typing_tasks:
            self.typing_tasks[chat_id].cancel()
        self.typing_tasks[chat_id] = asyncio.create_task(
            self._typing_loop(context.bot, chat_id)
        )
        
        try:
            # 获取对话历史
            history = self.user_history.get(user_id, [])
            
            # 如果没有 AI 接口，返回测试响应
            if not self.ai:
                await self._stop_typing(chat_id)
                response = self._get_test_response(user_message)
                await update.message.reply_text(response)
                self._update_history(user_id, user_message, response)
                return
            
            # 流式生成响应
            await self._handle_stream_generation(
                update=update,
                user_id=user_id,
                user_message=user_message,
                history=history
            )
            
        except Exception as e:
            logger.error(f"处理消息失败：{e}", exc_info=True)
            await self._stop_typing(chat_id)
            await update.message.reply_text(f"❌ 处理失败：{str(e)}")
    
    async def _stop_typing(self, chat_id: int):
        """停止打字状态"""
        if chat_id in self.typing_tasks:
            self.typing_tasks[chat_id].cancel()
            del self.typing_tasks[chat_id]
    
    async def _handle_stream_generation(
        self,
        update: Update,
        user_id: int,
        user_message: str,
        history: List[Dict[str, str]]
    ):
        """处理流式生成"""
        try:
            # 发送初始消息
            initial_message = await update.message.reply_text("🤔 思考中...")
            
            # 构建输入
            if history:
                context = "\n".join([
                    f"{h['role']}: {h['content']}"
                    for h in history[-self.max_context_length:]
                ])
                full_input = f"{context}\nUser: {user_message}\nAssistant:"
            else:
                full_input = f"User: {user_message}\nAssistant:"
            
            full_response = ""
            last_update_time = time.time()
            
            # 流式生成
            async for chunk in self.ai.generate_stream(full_input, max_tokens=200):
                full_response += chunk
                
                current_time = time.time()
                if current_time - last_update_time > 0.5 or len(full_response) > 4000:
                    last_update_time = current_time
                    
                    content_to_send = full_response[:4000]
                    
                    try:
                        await initial_message.edit_text(content_to_send + "▌")
                    except Exception:
                        pass
                
                if len(full_response) > 4000:
                    break
            
            # 完成
            await self._stop_typing(update.effective_chat.id)
            
            # 最终编辑
            try:
                final_text = full_response[:4090]
                await initial_message.edit_text(final_text)
            except Exception as e:
                logger.error(f"最终编辑失败：{e}")
            
            # 更新历史
            self._update_history(user_id, user_message, full_response)
            
            logger.info(f"回复用户 {user_id}: {full_response[:50]}...")
            
        except Exception as e:
            logger.error(f"流式生成失败：{e}", exc_info=True)
            await self._stop_typing(update.effective_chat.id)
            await update.message.reply_text(f"❌ 生成失败：{str(e)}")
    
    def _update_history(self, user_id: int, user_message: str, assistant_response: str):
        """更新对话历史"""
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        self.user_history[user_id].append({
            'role': 'user',
            'content': user_message
        })
        self.user_history[user_id].append({
            'role': 'assistant',
            'content': assistant_response
        })
        
        # 保持历史长度
        if len(self.user_history[user_id]) > self.max_context_length * 2:
            self.user_history[user_id] = self.user_history[user_id][-self.max_context_length * 2:]
    
    def _get_test_response(self, message: str) -> str:
        """测试响应 (无 AI 接口时使用)"""
        responses = {
            "你好": "你好！我是类人脑双系统AI助手。",
            "介绍": "我基于海马体-新皮层双系统架构，支持10ms高刷新推理和STDP在线学习。",
            "帮助": "直接发送消息即可与我对话！我支持流式输出。"
        }
        
        for key, value in responses.items():
            if key in message:
                return value
        
        return f"收到：{message}。请配置真实的AI模型接口以获得更好的响应。"
    
    def run(self):
        """运行 Bot"""
        logger.info("正在启动 Telegram Bot...")
        
        # 创建应用
        self.application = Application.builder().token(self.token).build()
        
        # 添加处理器
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # 启动 Bot
        logger.info("Bot 已启动，正在监听消息...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)


def create_bot(
    token: str,
    ai_interface=None,
    **kwargs
):
    """
    快捷创建 Bot 实例
    
    Args:
        token: Telegram Bot Token
        ai_interface: BrainAIInterface 实例
        **kwargs: 其他参数
    
    Returns:
        BrainAIBot 实例
    """
    return BrainAIBot(token=token, ai_interface=ai_interface, **kwargs)
