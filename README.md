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

**进行中**:
- 🔄 数据处理模块
- 🔄 交易环境模块
- 🔄 RL算法实现

**计划中**:
- ⏳ 策略执行模块
- ⏳ 风险管理模块
- ⏳ 回测引擎
- ⏳ 单元测试
- ⏳ 集成测试

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
