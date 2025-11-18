#!/usr/bin/env python3
"""
强化学习交易终端 - 项目初始化脚本
用于快速创建项目结构和基础文件
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 项目结构定义
PROJECT_STRUCTURE = {
    'rl_trading_system': {
        'core': ['__init__.py', 'base.py', 'config.py', 'exceptions.py'],
        'data': ['__init__.py', 'collector.py', 'preprocessor.py', 'feature_engineer.py', 'storage.py'],
        'environment': ['__init__.py', 'trading_env.py', 'market_sim.py', 'reward.py'],
        'agents': ['__init__.py', 'base_agent.py', 'dqn.py', 'ppo.py', 'sac.py'],
        'strategies': ['__init__.py', 'signals.py', 'portfolio.py', 'execution.py'],
        'risk': ['__init__.py', 'metrics.py', 'controls.py', 'monitor.py'],
        'backtest': ['__init__.py', 'engine.py', 'analyzer.py', 'report.py'],
        'utils': ['__init__.py', 'logger.py', 'metrics.py', 'validators.py'],
    },
    'tests': {
        'unit': ['__init__.py'],
        'integration': ['__init__.py'],
        'performance': ['__init__.py'],
    },
    'configs': [],
    'data': {
        'raw': [],
        'processed': [],
        'features': [],
    },
    'models': {
        'checkpoints': [],
        'trained': [],
    },
    'logs': [],
    'reports': [],
    'notebooks': [],
    'docs': [],
}

# 基础文件内容模板
FILE_TEMPLATES = {
    '__init__.py': '''"""
{module_path} module
"""

__version__ = "1.0.0"
''',
    
    'base.py': '''"""
Base classes for the trading system
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModule(ABC):
    """Abstract base class for all modules"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base module
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._validate_config()
        self._initialize()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        pass
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize module components"""
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Main processing method"""
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
''',
    
    'requirements.txt': '''# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Machine Learning
torch>=2.0.0
stable-baselines3>=2.0.0
gymnasium>=0.28.0

# Financial Libraries
TA-Lib>=0.4.24
empyrical>=0.5.5
pyfolio>=0.9.2

# Data Sources
ccxt>=4.0.0
yfinance>=0.2.18

# Database
influxdb-client>=1.36.0
redis>=4.5.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
click>=8.1.3
tqdm>=4.65.0

# Monitoring
prometheus-client>=0.16.0
tensorboard>=2.13.0
wandb>=0.15.0

# Testing
pytest>=7.3.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
''',
    
    'requirements-dev.txt': '''# Development dependencies
black>=23.0.0
flake8>=6.0.0
mypy>=1.3.0
isort>=5.12.0

# Testing
pytest-benchmark>=4.0.0
memory-profiler>=0.60.0
line-profiler>=4.0.0

# Documentation
sphinx>=6.2.0
sphinx-rtd-theme>=1.2.0
sphinx-autodoc-typehints>=1.23.0

# Jupyter
jupyter>=1.0.0
ipywidgets>=8.0.0
''',
    
    '.env.example': '''# API Keys (DO NOT COMMIT ACTUAL KEYS!)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here

# Database
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_token_here
INFLUXDB_ORG=your_org
INFLUXDB_BUCKET=trading_data

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Model Configuration
MODEL_PATH=./models/trained/
CHECKPOINT_PATH=./models/checkpoints/

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/trading.log

# Monitoring
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=rl-trading

# Risk Management
MAX_POSITION_SIZE=0.1
MAX_DRAWDOWN=0.2
STOP_LOSS=0.05
TAKE_PROFIT=0.15
''',
    
    'setup.py': '''"""
Setup configuration for RL Trading System
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rl-trading-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular reinforcement learning trading system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rl-trading-system",
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rl-trading=rl_trading_system.cli:main",
        ],
    },
)
''',
    
    'README.md': '''# 强化学习交易终端 (RL Trading System)

## 📖 概述
一个模块化、可扩展的强化学习交易系统框架，支持多种RL算法和交易策略。

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发依赖
```

### 配置环境
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

### 运行示例
```python
from rl_trading_system import TradingEnvironment, DQNAgent

# 创建环境
env = TradingEnvironment(config)

# 初始化智能体
agent = DQNAgent(env.observation_space, env.action_space)

# 训练
agent.train(env, episodes=1000)

# 回测
results = agent.backtest(env, start_date, end_date)
```

## 📁 项目结构
```
rl_trading_system/
├── core/           # 核心模块
├── data/           # 数据处理
├── environment/    # 交易环境
├── agents/         # RL算法
├── strategies/     # 交易策略
├── risk/          # 风险管理
├── backtest/      # 回测系统
└── utils/         # 工具函数
```

## 📚 文档
- [开发需求文档](./rl_trading_requirements.md)
- [开发规则文档](./rl_trading_development_guidelines.md)
- [模块说明文档](./module_documentation.md)
- [Claude Code指令](./claude_code_instructions.md)

## 🧪 测试
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_agents.py

# 生成覆盖率报告
pytest --cov=rl_trading_system --cov-report=html
```

## 📊 性能指标
- 数据处理延迟: < 10ms
- 模型推理延迟: < 5ms
- 回测速度: > 10,000 ticks/秒

## 🤝 贡献指南
1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 许可证
MIT License

## 👥 作者
- Claude Code Assistant
- [Your Name]

## 🙏 致谢
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [TensorTrade](https://github.com/tensortrade-org/tensortrade)
''',
    
    '.gitignore': '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Environment Variables
.env
.env.local
.env.*.local

# Logs
logs/
*.log

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/checkpoints/*
models/trained/*
!models/checkpoints/.gitkeep
!models/trained/.gitkeep

# Reports
reports/*.html
reports/*.pdf

# Testing
.coverage
htmlcov/
.pytest_cache/
.benchmarks/

# Documentation
docs/_build/
docs/_static/
docs/_templates/

# Monitoring
.tensorboard/
wandb/
mlruns/

# Temporary files
*.tmp
*.bak
*.swp
*~
''',
}


def create_directory_structure(base_path: Path, structure: dict, level: int = 0):
    """递归创建目录结构"""
    for name, content in structure.items():
        path = base_path / name
        
        if isinstance(content, dict):
            # 创建目录
            path.mkdir(exist_ok=True)
            print(f"{'  ' * level}📁 {name}/")
            # 递归创建子目录
            create_directory_structure(path, content, level + 1)
        elif isinstance(content, list):
            # 创建目录和文件
            if not path.exists():
                path.mkdir(exist_ok=True)
                print(f"{'  ' * level}📁 {name}/")
            
            for file in content:
                file_path = path / file
                if not file_path.exists():
                    # 获取模板内容
                    if file in FILE_TEMPLATES:
                        content = FILE_TEMPLATES[file].format(
                            module_path=name,
                            date=datetime.now().strftime("%Y-%m-%d")
                        )
                    else:
                        content = ""
                    
                    # 写入文件
                    file_path.write_text(content)
                    print(f"{'  ' * (level + 1)}📄 {file}")


def create_root_files(base_path: Path):
    """创建根目录文件"""
    root_files = [
        'requirements.txt',
        'requirements-dev.txt',
        '.env.example',
        'setup.py',
        'README.md',
        '.gitignore'
    ]
    
    for file in root_files:
        file_path = base_path / file
        if not file_path.exists():
            if file in FILE_TEMPLATES:
                file_path.write_text(FILE_TEMPLATES[file])
                print(f"📄 {file}")


def create_gitkeep_files(base_path: Path):
    """在空目录中创建.gitkeep文件"""
    empty_dirs = [
        'data/raw',
        'data/processed',
        'data/features',
        'models/checkpoints',
        'models/trained',
        'logs',
        'reports',
        'configs',
        'notebooks',
        'docs'
    ]
    
    for dir_path in empty_dirs:
        full_path = base_path / dir_path
        gitkeep_path = full_path / '.gitkeep'
        if not gitkeep_path.exists():
            gitkeep_path.write_text('')


def main():
    """主函数"""
    print("=" * 50)
    print("🚀 强化学习交易终端 - 项目初始化")
    print("=" * 50)
    
    # 获取项目路径
    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
    else:
        project_path = Path.cwd() / "rl_trading_project"
    
    # 创建项目根目录
    project_path.mkdir(exist_ok=True)
    print(f"\n📍 项目路径: {project_path}\n")
    
    # 创建目录结构
    print("创建目录结构:")
    create_directory_structure(project_path, PROJECT_STRUCTURE)
    
    # 创建根目录文件
    print("\n创建配置文件:")
    create_root_files(project_path)
    
    # 创建.gitkeep文件
    create_gitkeep_files(project_path)
    
    # 复制文档文件
    print("\n📚 复制文档文件...")
    docs_to_copy = [
        'rl_trading_requirements.md',
        'rl_trading_development_guidelines.md',
        'module_documentation.md',
        'claude_code_instructions.md'
    ]
    
    for doc in docs_to_copy:
        src = Path(doc)
        if src.exists():
            dst = project_path / doc
            dst.write_text(src.read_text())
            print(f"  ✓ {doc}")
    
    print("\n" + "=" * 50)
    print("✅ 项目初始化完成！")
    print("\n下一步:")
    print(f"1. cd {project_path}")
    print("2. python -m venv venv")
    print("3. source venv/bin/activate  # Linux/Mac")
    print("   或 venv\\Scripts\\activate  # Windows")
    print("4. pip install -r requirements.txt")
    print("5. cp .env.example .env")
    print("6. 编辑 .env 文件配置API密钥")
    print("\n祝您开发愉快！ 🎉")
    print("=" * 50)


if __name__ == "__main__":
    main()
