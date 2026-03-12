# LSDC 引擎

## 逻辑自相似稠密补齐 (Logic Self-similar Dense Completion) 引擎

### 核心数学原理

#### 1. 离散状态转移

```
S_n ──f()──> S_{n+1}
```

每个逻辑节点 `S_n` 代表一个离散的思维状态。推理过程是状态空间的离散转移。

#### 2. 窄宽带补齐

```
f(S_n, Δt) → S_{n+1}
```

**核心创新**：每次只喂入上一个微步的 `Conclusion` 和当前的 `Goal`，丢弃所有历史过程。

这解决了小模型（如0.8B）的上下文过载问题，强制模型专注于当前推理步。

#### 3. 自相似结构

```
[前提, 推演, 结论] 三位一体
```

**关键特性**：任意尺度的逻辑结构保持同构。

- **微观尺度**：单个推理步
- **宏观尺度**：完整推理链
- **结构同构**：每个节点都包含 [前提, 推演, 结论] 结构

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        LSDC 引擎                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Goal      │───>│  Narrow     │───>│   Logic     │    │
│  │   输入      │    │  Bandwidth  │    │   Node      │    │
│  │             │    │  Filter     │    │   S_n       │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│         │                  │                  │            │
│         │                  │                  │            │
│         │                  v                  v            │
│         │           ┌─────────────┐    ┌─────────────┐    │
│         │           │  Previous   │    │   Dense     │    │
│         │           │  Conclusion │    │   Check     │    │
│         │           │  (丢弃历史) │    │   稠密性检查│    │
│         │           └─────────────┘    └─────────────┘    │
│         │                                     │            │
│         │                                     │            │
│         v                                     v            │
│  ┌─────────────┐                       ┌─────────────┐    │
│  │   Model     │<──────────────────────│   补齐      │    │
│  │   DeepSeek  │                       │   Densify   │    │
│  │   R1 1.5B   │                       │             │    │
│  └─────────────┘                       └─────────────┘    │
│         │                                     │            │
│         v                                     v            │
│  ┌─────────────┐                       ┌─────────────┐    │
│  │   Output    │                       │   Logic     │    │
│  │   微步生成  │──────────────────────>│   Chain     │    │
│  │   (16-32tok)│                       │   逻辑链    │    │
│  └─────────────┘                       └─────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 快速开始

#### 1. 安装依赖

```bash
cd lsdc_engine
bash setup.sh
```

#### 2. 下载模型

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', local_dir='../models/DeepSeek-R1-Distill-Qwen-1.5B')"
```

#### 3. 运行

**交互模式：**
```bash
python3 main.py --mode interactive
```

**服务器模式：**
```bash
python3 main.py --mode server --port 8000
```

**测试模式：**
```bash
python3 main.py --mode test --goal "3月份20天房租1600元，月租是多少？"
```

### API 接口

#### POST /process

处理问题（非流式）

```json
{
  "goal": "月租是多少？",
  "context": "3月份20天房租1600元",
  "max_iterations": 10
}
```

#### POST /stream

流式处理（SSE）

```javascript
const eventSource = new EventSource('/stream');
eventSource.addEventListener('node', (e) => {
  console.log(JSON.parse(e.data));
});
```

### 核心配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_new_tokens | 24 | 微步生成长度（高刷新） |
| window_size | 2 | 窄宽带窗口大小 |
| do_sample | False | 使用Greedy Search |
| max_iterations | 20 | 最大迭代次数 |

### 文件结构

```
lsdc_engine/
├── main.py              # 主入口
├── logic_processor.py   # 自相似逻辑核心算法
├── model_handler.py     # 模型加载与窄宽带控制
├── app.py               # FastAPI 流式接口
├── setup.sh             # 环境安装脚本
├── requirements.txt     # 依赖清单
└── README.md            # 本文档
```

### 数学推导

#### 逻辑稠密度定义

```
D(S_n) = |推理步骤| / |逻辑跨度|
```

- `|推理步骤|`：显式推理步骤数量
- `|逻辑跨度|`：从前提到结论的概念距离

当 `D(S_n) < θ` 时，触发补齐操作。

#### 状态转移方程

```
S_{n+1} = S_n ⊕ δ

其中:
- δ = f(S_n, Goal) 是逻辑增量
- ⊕ 是状态融合算子
```

### 许可证

MIT License
