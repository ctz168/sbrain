#!/usr/bin/env python3
"""
LSDC 引擎 - FastAPI 流式接口

核心特性：
1. SSE (Server-Sent Events) 实时推送
2. 状态连续保持
3. 流式输出稠密补齐结果
"""

import os
import sys
import json
import asyncio
from typing import Optional, Dict, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsdc_engine.logic_processor import LogicProcessor, LogicNode, create_logic_processor
from lsdc_engine.model_handler import create_model_handler


# ============================================================
# 请求/响应模型
# ============================================================

class ProcessRequest(BaseModel):
    """处理请求"""
    goal: str                    # 目标问题
    context: Optional[str] = None  # 上下文
    max_iterations: int = 10      # 最大迭代次数


class NodeResponse(BaseModel):
    """节点响应"""
    node_id: int
    phase: str
    premise: str
    derivation: str
    conclusion: str
    density: float
    timestamp: str


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="LSDC 引擎",
    description="逻辑自相似稠密补齐引擎",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局逻辑处理器
logic_processor: Optional[LogicProcessor] = None

# 状态缓存（内存字典，可替换为Redis）
state_cache: Dict[str, Dict] = {}


@app.on_event("startup")
async def startup_event():
    """启动事件：初始化模型"""
    global logic_processor
    
    print("\n" + "="*60)
    print("LSDC 引擎启动")
    print("="*60)
    
    # 初始化逻辑处理器
    model_path = os.environ.get("MODEL_PATH", "../models/Qwen3.5-0.8B")
    device = os.environ.get("DEVICE", "cpu")
    
    print(f"\n初始化逻辑处理器...")
    print(f"模型路径: {model_path}")
    print(f"设备: {device}")
    
    logic_processor = create_logic_processor(model_path, device)
    
    print("\n" + "="*60)
    print("✓ LSDC 引擎就绪")
    print("="*60)


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "LSDC 引擎",
        "version": "1.0.0",
        "description": "逻辑自相似稠密补齐引擎",
        "endpoints": {
            "/process": "POST - 处理问题",
            "/stream": "POST - 流式处理",
            "/state/{session_id}": "GET - 获取状态",
            "/health": "GET - 健康检查"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": logic_processor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/process")
async def process(request: ProcessRequest):
    """
    处理问题（非流式）
    
    返回完整的逻辑链
    """
    if not logic_processor:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 处理
    nodes = []
    for node in logic_processor.process(request.goal, request.context):
        nodes.append(NodeResponse(
            node_id=node.node_id,
            phase=node.phase.value,
            premise=node.premise,
            derivation=node.derivation,
            conclusion=node.conclusion,
            density=node.density,
            timestamp=datetime.now().isoformat()
        ))
    
    # 获取逻辑链
    chain = logic_processor.get_chain()
    chain_text = chain.to_text() if chain else ""
    
    return {
        "goal": request.goal,
        "nodes": [n.dict() for n in nodes],
        "chain_text": chain_text,
        "total_nodes": len(nodes)
    }


@app.post("/stream")
async def stream_process(request: ProcessRequest):
    """
    流式处理
    
    使用SSE实时推送每个逻辑节点
    """
    if not logic_processor:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    async def event_generator():
        """SSE事件生成器"""
        session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 发送会话开始
        yield f"event: start\ndata: {json.dumps({'session_id': session_id, 'goal': request.goal})}\n\n"
        
        # 处理每个节点
        for node in logic_processor.process(request.goal, request.context):
            # 构建节点数据
            node_data = {
                "node_id": node.node_id,
                "phase": node.phase.value,
                "premise": node.premise,
                "derivation": node.derivation,
                "conclusion": node.conclusion,
                "density": node.density,
                "timestamp": datetime.now().isoformat()
            }
            
            # 发送节点事件
            yield f"event: node\ndata: {json.dumps(node_data)}\n\n"
            
            # 缓存状态
            if session_id not in state_cache:
                state_cache[session_id] = {"nodes": [], "goal": request.goal}
            state_cache[session_id]["nodes"].append(node_data)
            
            # 短暂延迟
            await asyncio.sleep(0.1)
        
        # 获取逻辑链
        chain = logic_processor.get_chain()
        chain_text = chain.to_text() if chain else ""
        
        # 发送完成事件
        yield f"event: complete\ndata: {json.dumps({'session_id': session_id, 'chain_text': chain_text, 'total_nodes': len(state_cache.get(session_id, {}).get('nodes', []))})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.get("/state/{session_id}")
async def get_state(session_id: str):
    """获取会话状态"""
    if session_id not in state_cache:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return state_cache[session_id]


@app.delete("/state/{session_id}")
async def clear_state(session_id: str):
    """清除会话状态"""
    if session_id in state_cache:
        del state_cache[session_id]
    return {"status": "cleared"}


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
