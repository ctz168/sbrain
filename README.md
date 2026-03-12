# 类人脑双系统全闭环AI架构 (sbrain)

基于 DeepSeek-R1-Distill-Qwen-1.5B 底座模型的类人脑AI架构，实现端侧设备上极高的智能水平。

## 核心特性

### 刚性约束（100%严格遵守）
- ✅ **底座唯一约束**：仅使用 DeepSeek-R1-Distill-Qwen-1.5B 单模型
- ✅ **权重安全约束**：90%静态权重冻结 + 10%动态权重STDP更新
- ✅ **端侧算力约束**：INT4量化后显存占用极低，单周期算力开销≤10%
- ✅ **原生执行范式**：10ms/100Hz刷新周期，O(1)注意力复杂度
- ✅ **学习机制约束**：纯STDP学习，无反向传播
- ✅ **全闭环约束**：所有模块深度耦合，无外挂

### 核心模块

#### 模块1：双轨权重原生改造
- 权重按9:1拆分为静态分支与动态分支
- 静态分支：90%冻结，继承官方预训练权重
- 动态分支：10%可更新，仅通过STDP规则
- 前向融合：9:1加权融合输出

#### 模块2：多尺度时序嵌套推理引擎
- 10ms/100Hz原生执行周期
- 三通路认知：直觉通路、逻辑推理、深度反思
- 层级化动态锚点O(1)注意力机制

#### 模块3：全链路STDP学习系统
- LTP/LTD公式驱动权重更新
- 全链路覆盖：注意力层、FFN层、海马体门控、自评判
- 动态学习率适配

#### 模块4：元认知双闭环校验系统
- 元认知特征提取：注意力熵、STDP激活、语义一致性
- 在线实时校验闭环
- 离线反思闭环

#### 模块5：海马体-新皮层协同记忆系统
- 内嗅皮层EC：特征编码（1536维→64维）
- 齿状回DG：模式分离
- CA3区：情景记忆库（1024条）
- CA1区：时序门控
- 长期语义记忆库（4096条三元组）
- 尖波涟漪SWR：离线记忆巩固

#### 模块6：多任务场景自适应
- 6大场景自动识别
- 场景专属预适配
- 在线场景优化

#### 模块7：多维度测评体系
- 记忆能力测评（40%）
- 推理能力测评（30%）
- 可靠性测评（15%）
- 端侧性能测评（10%）
- 学习能力测评（5%）

## 安装

```bash
# 克隆仓库
git clone https://github.com/ctz168/sbrain.git
cd sbrain

# 安装依赖
pip install -r requirements.txt
pip install python-telegram-bot

# 下载模型
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./models/DeepSeek-R1-Distill-Qwen-1.5B
```

## 快速开始

### Telegram Bot 对话 (DeepSeek-R1-1.5B)

```bash
python telegram_bot/run_bot_deepseek.py --token YOUR_BOT_TOKEN --model_path ./models/DeepSeek-R1-Distill-Qwen-1.5B
```

### 交互式对话

```bash
python main.py --mode chat --model-path ./models/DeepSeek-R1-Distill-Qwen-1.5B
```

### 训练

```bash
python main.py --mode train --model-path ./models/DeepSeek-R1-Distill-Qwen-1.5B
```

### 测评

```bash
python main.py --mode eval --model-path ./models/DeepSeek-R1-Distill-Qwen-1.5B
```

## 项目结构

```
sbrain/
├── configs/           # 配置文件
│   └── config.py      # 全局配置
├── core/              # 核心模块
│   ├── model.py       # 主模型
│   ├── deepseek_bot.py # DeepSeek Bot核心
│   └── dual_weight.py # 双轨权重
├── telegram_bot/      # Telegram Bot
│   └── run_bot_deepseek.py # DeepSeek Bot启动脚本
├── stdp/              # STDP学习系统
├── hippocampus/       # 海马体系统
├── inference/         # 推理引擎
├── metacognition/     # 元认知系统
├── scene_adapt/       # 场景适配
├── evaluation/        # 测评系统
├── main.py            # 主入口
├── requirements.txt   # 依赖
└── README.md          # 说明文档
```

## 许可证

MIT License

## 联系方式

- GitHub: https://github.com/ctz168/sbrain
```
