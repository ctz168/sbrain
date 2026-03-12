# 类人脑双系统全闭环AI架构 (sbrain)

基于DeepSeek-R1-Distill-Qwen-1.5B底座模型的类人脑AI架构，实现端侧设备上接近13B模型的智能水平。

## 核心特性

### 刚性约束（100%严格遵守）
- ✅ **底座唯一约束**：仅使用DeepSeek-R1-Distill-Qwen-1.5B单模型
- ✅ **权重安全约束**：90%静态权重冻结 + 10%动态权重STDP更新
- ✅ **端侧算力约束**：INT4量化后显存≤420MB，单周期算力开销≤10%
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
- 内嗅皮层EC：特征编码（768维→64维）
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

# 下载模型
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./models/DeepSeek-R1-Distill-Qwen-1.5B
```

## 快速开始

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

### 服务模式

```bash
python main.py --mode serve --model-path ./models/DeepSeek-R1-Distill-Qwen-1.5B
```

## 使用示例

```python
from sbrain import BrainAIModel, create_model

# 创建模型
model = create_model(model_path='./models/DeepSeek-R1-Distill-Qwen-1.5B')

# 对话
response = model.chat("你好，请介绍一下你自己")
print(response)

# 获取统计信息
stats = model.get_stats()
print(stats)

# 保存检查点
model.save_checkpoint('my_checkpoint.pt')
```

## 项目结构

```
sbrain/
├── configs/           # 配置文件
│   └── config.py      # 全局配置
├── core/              # 核心模块
│   ├── model.py       # 主模型
│   └── dual_weight.py # 双轨权重
├── stdp/              # STDP学习系统
│   └── stdp_engine.py # STDP引擎
├── hippocampus/       # 海马体系统
│   └── hippocampus_system.py
├── inference/         # 推理引擎
│   └── engine.py
├── metacognition/     # 元认知系统
│   └── metacognition_system.py
├── scene_adapt/       # 场景适配
│   └── scene_system.py
├── evaluation/        # 测评系统
│   └── evaluator.py
├── training/          # 训练脚本
├── deployment/        # 部署脚本
├── tests/             # 测试文件
├── docs/              # 文档
├── main.py            # 主入口
├── requirements.txt   # 依赖
└── README.md          # 说明文档
```

## 测评指标

| 维度 | 指标 | 目标 |
|------|------|------|
| 记忆能力 | 100k token保持率 | ≥95% |
| 记忆能力 | 记忆混淆率 | ≤1% |
| 记忆能力 | 跨会话召回率 | ≥90% |
| 记忆能力 | 抗遗忘能力 | ≥99% |
| 推理能力 | GSM8K准确率 | ≥40% |
| 推理能力 | HumanEval准确率 | ≥25% |
| 推理能力 | CommonsenseQA准确率 | ≥60% |
| 可靠性 | 事实准确率 | ≥90% |
| 可靠性 | 幻觉率 | ≤8% |
| 性能 | 显存占用 | ≤420MB |
| 性能 | 推理延迟 | ≤20ms |
| 学习能力 | 学习速度提升 | ≥400% |

## 端侧部署

### 树莓派4B

```bash
# 安装依赖
pip install -r requirements.txt

# 运行
python main.py --mode chat --device cpu --quantization INT4
```

### 安卓手机

```bash
# 使用Termux运行
pkg install python
pip install -r requirements.txt
python main.py --mode chat
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

- GitHub: https://github.com/ctz168/sbrain
