# 强化学习交易终端 (RL Trading System)

## 📖 概述

一个模块化、可扩展的强化学习交易系统框架，支持多种RL算法和交易策略。

## 🏗️ 系统架构

```
rl_trading_system/
├── core/                  # 核心模块
│   ├── base.py           # 基类定义
│   ├── config.py         # 配置管理
│   └── exceptions.py     # 异常定义
├── data/                  # 数据处理
├── environment/           # 交易环境
├── agents/                # RL智能体
├── strategies/            # 交易策略
├── risk/                  # 风险管理
├── backtest/              # 回测系统
└── utils/                 # 工具函数
    ├── logger.py         # 日志工具
    └── validators.py     # 验证工具
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from rl_trading_system.core import TradingConfig

# 创建配置
config = TradingConfig()
config.data.symbols = ['BTC/USDT', 'ETH/USDT']
config.model.learning_rate = 0.001

# 保存配置
config.save_to_file('config.json')

# 加载配置
config = TradingConfig.load_from_file('config.json')
```

## 📚 文档

- [开发需求文档](./RL_trading_requirements.md)
- [开发规则文档](./rl_trading_development_guidelines.md)
- [模块说明文档](./module_documentation.md)
- [Claude Code指令](./claude_code_instructions.md)

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_core.py

# 生成覆盖率报告
pytest --cov=rl_trading_system --cov-report=html
```

## 📊 已实现功能

### ✅ 核心模块 (core)
- ✅ BaseModule - 所有模块的抽象基类
- ✅ BaseStrategy - 策略基类
- ✅ BaseAgent - 智能体基类
- ✅ TradingConfig - 完整的配置管理系统
- ✅ 异常层级系统

### ✅ 交易环境 (environment) 🆕
- ✅ **StockTradingEnv** - FinRL风格的多资产交易环境
  - 支持离散和连续动作空间
  - 完整的交易成本建模
  - 技术指标集成
  - 组合价值跟踪
- ✅ **SharpeRewardEnv** - Sharpe Ratio优化环境
  - 风险调整收益优化
  - 滚动窗口Sharpe计算
  - 适应性强的奖励函数

### ✅ RL智能体 (agents) 🆕
- ✅ **DQNAgent** - Deep Q-Network智能体
  - Experience Replay
  - Target Network
  - Double DQN (可选)
  - Dueling DQN (可选)
  - Epsilon-greedy探索
  - PyTorch实现

### ✅ 数据处理 (data) 🆕
- ✅ **FeatureEngineer** - 技术指标工程
  - 10种技术指标实现
  - SMA, EMA, RSI, MACD, Bollinger Bands
  - ATR, CCI, Williams %R, OBV, VWAP
  - 纯NumPy/Pandas实现（无外部依赖）

### ✅ 工具模块 (utils)
- ✅ 统一日志系统
  - 文件日志
  - 控制台日志
  - JSON结构化日志
  - 性能日志
  - 审计日志
- ✅ 数据验证工具
  - OHLCV数据验证
  - 特征验证
  - 订单验证
  - 持仓验证
  - 配置验证
  - 状态和动作验证

### ✅ 示例与案例 (examples) 🆕
- ✅ **案例1**: FinRL风格多资产交易 (DQN)
- ✅ **案例2**: 基于技术指标的DQN交易
- ✅ **案例3**: TDQN Sharpe Ratio优化

## 🔧 配置示例

```python
from rl_trading_system.core import TradingConfig

config = TradingConfig()

# 数据配置
config.data.source = "binance"
config.data.symbols = ["BTC/USDT", "ETH/USDT"]
config.data.timeframe = "1h"

# 模型配置
config.model.model_type = "dqn"
config.model.learning_rate = 0.001
config.model.batch_size = 32

# 风险配置
config.risk.max_position_size = 0.1
config.risk.max_drawdown = 0.2
config.risk.stop_loss = 0.05

# 环境配置
config.environment.initial_capital = 10000.0
config.environment.commission = 0.001
```

## 🎯 三大经典案例实现

基于学术论文和开源项目的经典RL交易案例：

### 案例1: FinRL风格多资产交易
**来源**: NeurIPS 2020, AI4Finance-Foundation/FinRL
**特点**:
- 多资产组合管理
- 完整的技术指标体系
- 灵活的动作空间（离散/连续）
- 交易成本和滑点建模

**运行示例**:
```bash
python examples/case1_finrl_trading.py
```

### 案例2: 基于技术指标的DQN
**来源**: arXiv:2304.06037
**特点**:
- 简洁的状态设计
- 技术指标驱动（MA, RSI, MACD）
- 离散动作空间
- 直接收益优化

**运行示例**:
```bash
python examples/case2_dqn_technical_indicators.py
```

### 案例3: TDQN Sharpe Ratio优化
**来源**: arXiv:2004.06627
**特点**:
- Sharpe Ratio作为奖励
- 风险调整收益优化
- 滚动窗口统计
- 适应不同市场环境

**运行示例**:
```bash
python examples/case3_tdqn_sharpe_optimization.py
```

详细实现文档请查看: [IMPLEMENTATION_REPORT.md](./IMPLEMENTATION_REPORT.md)

## 📝 开发状态

**当前版本**: 1.0.0

**已完成**:
- ✅ 项目结构初始化
- ✅ 核心模块实现
- ✅ 工具模块实现
- ✅ 配置管理系统
- ✅ 异常处理系统
- ✅ 日志系统
- ✅ 验证工具
- ✅ 交易环境模块 (2个环境)
- ✅ DQN智能体实现
- ✅ 数据处理和特征工程
- ✅ 三个经典案例复现 (2400+行代码)

**计划中**:
- ⏳ PPO和SAC算法实现
- ⏳ 策略执行模块
- ⏳ 风险管理模块
- ⏳ 完整回测引擎
- ⏳ 真实数据集成
- ⏳ 单元测试扩展

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat(module): add new feature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 许可证

MIT License

## 👥 作者

- Claude Code Assistant

## 🙏 致谢

- [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [TensorTrade](https://github.com/tensortrade-org/tensortrade)
