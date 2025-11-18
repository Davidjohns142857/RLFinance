# 强化学习交易终端 - 高质量开发规则文档

## 1. 开发原则

### 1.1 核心原则
1. **模块化优先**: 每个功能必须可独立测试和替换
2. **领域驱动设计**: 代码结构应反映交易领域的概念和流程
3. **性能敏感**: 交易系统对延迟极其敏感，优化是必需的
4. **失败安全**: 任何错误都不应导致资金损失
5. **可追溯性**: 所有交易决策必须可审计和重现

### 1.2 开发流程
```mermaid
graph LR
    A[需求分析] --> B[文献调研]
    B --> C[原型设计]
    C --> D[模块开发]
    D --> E[单元测试]
    E --> F[集成测试]
    F --> G[性能优化]
    G --> H[文档编写]
    H --> I[代码审查]
```

## 2. 编码标准

### 2.1 Python编码规范
```python
# 文件头部模板
"""
Module: {module_name}
Description: {brief_description}
Author: Claude Code
Date: {date}
Version: {version}

Dependencies:
    - {dependency_1}
    - {dependency_2}

Usage Example:
    >>> from module import Class
    >>> instance = Class(config)
    >>> result = instance.method()
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 配置日志
logger = logging.getLogger(__name__)
```

### 2.2 命名规范
- **类名**: PascalCase (例: `TradingEnvironment`)
- **函数名**: snake_case (例: `calculate_sharpe_ratio`)
- **常量**: UPPER_SNAKE_CASE (例: `MAX_POSITION_SIZE`)
- **私有方法**: 前缀下划线 (例: `_validate_order`)
- **配置参数**: 使用dataclass或pydantic

### 2.3 类型注解要求
```python
# 必须使用完整的类型注解
def process_market_data(
    data: pd.DataFrame,
    features: List[str],
    lookback: int = 20,
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    处理市场数据并提取特征
    
    Args:
        data: OHLCV数据DataFrame
        features: 要计算的特征列表
        lookback: 回望期窗口
        normalize: 是否归一化
    
    Returns:
        features_array: 特征数组
        stats: 统计信息字典
    
    Raises:
        ValueError: 数据格式错误
        KeyError: 缺少必要列
    """
    pass
```

## 3. 模块化设计规则

### 3.1 模块结构模板
```python
# base_module.py
class BaseModule(ABC):
    """抽象基类，定义接口"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._validate_config()
        self._initialize()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """验证配置参数"""
        pass
    
    @abstractmethod
    def _initialize(self) -> None:
        """初始化模块"""
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """主处理方法"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
```

### 3.2 工厂模式实现
```python
# module_factory.py
class ModuleFactory:
    """模块工厂，支持动态注册和创建"""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器：注册模块"""
        def decorator(module_class):
            cls._registry[name] = module_class
            return module_class
        return decorator
    
    @classmethod
    def create(cls, name: str, config: Dict):
        """创建模块实例"""
        if name not in cls._registry:
            raise ValueError(f"Unknown module: {name}")
        return cls._registry[name](config)

# 使用示例
@ModuleFactory.register("dqn")
class DQNStrategy(BaseStrategy):
    pass
```

## 4. 金融领域知识集成

### 4.1 因子工程规范
```python
class FactorEngineer:
    """因子工程基类"""
    
    # 必须实现的技术指标
    REQUIRED_INDICATORS = [
        'sma', 'ema', 'rsi', 'macd', 'bollinger_bands',
        'atr', 'adx', 'volume_profile', 'vwap'
    ]
    
    # 市场微观结构因子
    MICROSTRUCTURE_FACTORS = [
        'bid_ask_spread', 'order_imbalance', 'price_impact',
        'quote_stuffing_index', 'trade_intensity'
    ]
    
    def create_factor(
        self,
        name: str,
        formula: str,
        dependencies: List[str]
    ) -> Factor:
        """
        创建自定义因子
        
        Example:
            factor = create_factor(
                name="momentum_quality",
                formula="momentum * (1 - volatility/returns)",
                dependencies=["momentum", "volatility", "returns"]
            )
        """
        pass
```

### 4.2 风险指标计算
```python
@dataclass
class RiskMetrics:
    """风险指标数据类"""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    information_ratio: float
    calmar_ratio: float
    
    def to_dict(self) -> Dict:
        """转换为字典，便于序列化"""
        return asdict(self)
    
    def validate(self) -> bool:
        """验证指标合理性"""
        return all([
            -10 < self.sharpe_ratio < 10,
            0 < self.max_drawdown < 1,
            self.var_95 > 0,
            self.cvar_95 > self.var_95
        ])
```

## 5. 搜索和验证规则

### 5.1 信息源优先级
```python
TRUSTED_SOURCES = {
    "research": [
        "arxiv.org",  # 学术论文
        "papers.ssrn.com",  # 金融研究
        "jmlr.org",  # 机器学习期刊
    ],
    "implementation": [
        "github.com/quantopian",
        "github.com/hudson-and-thames",
        "github.com/AmazingAng/WTF-Solidity",
        "stable-baselines3.readthedocs.io"
    ],
    "market_data": [
        "finance.yahoo.com",
        "www.alphavantage.co",
        "polygon.io"
    ]
}

def verify_implementation(method_name: str) -> bool:
    """
    验证实现方法是否有可信来源支持
    
    Steps:
        1. 搜索学术论文验证理论基础
        2. 查找开源实现参考
        3. 对比多个实现确认正确性
    """
    pass
```

### 5.2 代码引用规范
```python
def calculate_feature(data: pd.DataFrame) -> pd.Series:
    """
    计算特征
    
    References:
        - Paper: "Advances in Financial Machine Learning" (Lopez de Prado, 2018)
        - Implementation: https://github.com/hudson-and-thames/mlfinlab
        - Formula: feature = log(high/low) * sqrt(volume)
    
    Verified:
        - Backtested on S&P500 (2010-2020): Sharpe 1.82
        - Cross-validated with 5-fold: Mean accuracy 0.67
    """
    pass
```

## 6. 文档维护规则

### 6.1 模块文档模板
```markdown
# Module: {ModuleName}

## Overview
Brief description of the module purpose and functionality

## Dependencies
- dependency1==version
- dependency2>=version

## API Reference

### Classes

#### `ClassName`
**Description**: What this class does

**Attributes**:
- `attribute1` (type): Description
- `attribute2` (type): Description

**Methods**:
- `method1(param1: type) -> return_type`: Description
- `method2(param1: type) -> return_type`: Description

### Functions

#### `function_name(param1: type, param2: type) -> return_type`
**Description**: What this function does

**Parameters**:
- `param1`: Description and constraints
- `param2`: Description and constraints

**Returns**:
- Description of return value

**Example**:
```python
result = function_name(value1, value2)
```

## Data Structures

### `DataStructureName`
```python
{
    'field1': type,  # Description
    'field2': type,  # Description
}
```

## Usage Examples

### Basic Usage
```python
# Code example
```

### Advanced Usage
```python
# Code example
```

## Performance Notes
- Time complexity: O(n)
- Space complexity: O(1)
- Optimization tips

## Change Log
- v1.0.0: Initial release
- v1.0.1: Bug fix in method X
```

### 6.2 函数文档规范
```python
def complex_calculation(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int = 20,
    method: str = 'exponential'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    执行复杂计算
    
    详细描述函数的功能和算法原理。
    
    Parameters
    ----------
    prices : np.ndarray
        价格数组，shape=(n_samples, n_assets)
    volumes : np.ndarray  
        成交量数组，shape=(n_samples, n_assets)
    window : int, optional
        滑动窗口大小，默认20
    method : str, optional
        计算方法，可选'exponential'或'simple'，默认'exponential'
    
    Returns
    -------
    signals : np.ndarray
        交易信号数组，shape=(n_samples, n_assets)
    metadata : Dict[str, Any]
        包含计算过程的元数据
        - 'computation_time': float
        - 'n_signals': int
        - 'signal_strength': List[float]
    
    Raises
    ------
    ValueError
        当prices和volumes形状不匹配时
    TypeError
        当method不在支持的方法列表中时
    
    See Also
    --------
    simple_calculation : 简化版本的计算
    validate_inputs : 输入验证函数
    
    Notes
    -----
    该函数实现了论文 [1]_ 中描述的算法。
    时间复杂度为 O(n * window * n_assets)。
    
    References
    ----------
    .. [1] Author, Title, Journal, 2023.
    
    Examples
    --------
    >>> prices = np.random.randn(100, 5)
    >>> volumes = np.random.randn(100, 5)
    >>> signals, meta = complex_calculation(prices, volumes, window=10)
    >>> signals.shape
    (100, 5)
    """
    # 输入验证
    if prices.shape != volumes.shape:
        raise ValueError(f"Shape mismatch: {prices.shape} != {volumes.shape}")
    
    if method not in ['exponential', 'simple']:
        raise TypeError(f"Unsupported method: {method}")
    
    # 主要计算逻辑
    # ...
    
    return signals, metadata
```

## 7. 测试规则

### 7.1 测试覆盖要求
- 单元测试覆盖率 > 90%
- 集成测试覆盖主要业务流程
- 性能测试验证延迟要求
- 回测验证策略有效性

### 7.2 测试模板
```python
# test_module.py
import pytest
import numpy as np
from unittest.mock import Mock, patch

class TestTradingModule:
    """交易模块测试类"""
    
    @pytest.fixture
    def setup_data(self):
        """准备测试数据"""
        return {
            'prices': np.random.randn(100, 5),
            'volumes': np.random.randn(100, 5)
        }
    
    def test_normal_case(self, setup_data):
        """测试正常情况"""
        result = module.process(setup_data)
        assert result is not None
        assert len(result) == 100
    
    def test_edge_case(self):
        """测试边界情况"""
        with pytest.raises(ValueError):
            module.process([])
    
    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6)
    ])
    def test_parametrized(self, input, expected):
        """参数化测试"""
        assert module.double(input) == expected
    
    @patch('module.external_api')
    def test_with_mock(self, mock_api):
        """使用Mock测试外部依赖"""
        mock_api.return_value = {'status': 'success'}
        result = module.call_api()
        assert result['status'] == 'success'
```

### 7.3 性能基准测试
```python
# benchmark.py
import time
import memory_profiler

class PerformanceBenchmark:
    """性能基准测试"""
    
    @staticmethod
    def measure_latency(func, *args, **kwargs):
        """测量函数延迟"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        latency = (time.perf_counter() - start) * 1000  # ms
        
        return result, latency
    
    @staticmethod
    @memory_profiler.profile
    def measure_memory(func, *args, **kwargs):
        """测量内存使用"""
        return func(*args, **kwargs)
    
    def assert_performance(self, func, max_latency_ms=10):
        """断言性能要求"""
        _, latency = self.measure_latency(func)
        assert latency < max_latency_ms, f"Latency {latency}ms exceeds limit"
```

## 8. 错误处理规则

### 8.1 异常层级设计
```python
# exceptions.py
class TradingException(Exception):
    """交易系统基础异常"""
    pass

class DataException(TradingException):
    """数据相关异常"""
    pass

class ExecutionException(TradingException):
    """执行相关异常"""
    pass

class RiskException(TradingException):
    """风险相关异常"""
    
    def __init__(self, message, position=None, risk_metrics=None):
        super().__init__(message)
        self.position = position
        self.risk_metrics = risk_metrics
        self._log_risk_event()
    
    def _log_risk_event(self):
        """记录风险事件到审计日志"""
        pass
```

### 8.2 错误恢复机制
```python
from functools import wraps
import backoff

def resilient_execution(max_retries=3, backoff_factor=2):
    """弹性执行装饰器"""
    def decorator(func):
        @wraps(func)
        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=max_retries,
            factor=backoff_factor
        )
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                # 保存现场
                save_context(args, kwargs, e)
                # 尝试恢复
                if can_recover(e):
                    return recover_execution(func, args, kwargs)
                raise
        return wrapper
    return decorator
```

## 9. 性能优化规则

### 9.1 数据处理优化
```python
# 使用向量化操作替代循环
# Bad
def calculate_returns_loop(prices):
    returns = []
    for i in range(1, len(prices)):
        returns.append((prices[i] - prices[i-1]) / prices[i-1])
    return returns

# Good
def calculate_returns_vectorized(prices):
    return np.diff(prices, axis=0) / prices[:-1]

# 使用numba加速计算密集型函数
from numba import jit, prange

@jit(nopython=True, parallel=True)
def fast_computation(data):
    result = np.zeros_like(data)
    for i in prange(len(data)):
        result[i] = complex_calculation(data[i])
    return result
```

### 9.2 内存优化
```python
# 使用生成器处理大数据
def process_large_dataset(file_path):
    """使用生成器逐行处理，避免一次性加载"""
    with open(file_path, 'r') as f:
        for line in f:
            yield process_line(line)

# 使用内存池
from multiprocessing import Pool

class MemoryEfficientProcessor:
    def __init__(self):
        self.pool = Pool(processes=4)
        self.chunk_size = 1000
    
    def process_data(self, data):
        chunks = [data[i:i+self.chunk_size] 
                  for i in range(0, len(data), self.chunk_size)]
        results = self.pool.map(self._process_chunk, chunks)
        return np.concatenate(results)
```

## 10. 部署和监控规则

### 10.1 容器化要求
```dockerfile
# Dockerfile template
FROM python:3.9-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 运行应用
CMD ["python", "main.py"]
```

### 10.2 监控指标
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 定义监控指标
trade_counter = Counter('trades_total', 'Total number of trades')
trade_latency = Histogram('trade_latency_seconds', 'Trade execution latency')
position_value = Gauge('position_value_usd', 'Current position value in USD')
model_confidence = Histogram('model_confidence', 'Model prediction confidence')

class MetricsCollector:
    """指标收集器"""
    
    @staticmethod
    def record_trade(symbol, action, latency):
        trade_counter.labels(symbol=symbol, action=action).inc()
        trade_latency.observe(latency)
    
    @staticmethod
    def update_position(value):
        position_value.set(value)
```

## 11. 安全规则

### 11.1 API密钥管理
```python
# 使用环境变量和密钥管理服务
import os
from cryptography.fernet import Fernet

class SecureConfig:
    """安全配置管理"""
    
    def __init__(self):
        self.cipher = Fernet(os.environ['ENCRYPTION_KEY'].encode())
    
    def get_api_key(self, exchange):
        """获取加密的API密钥"""
        encrypted = os.environ[f'{exchange.upper()}_API_KEY']
        return self.cipher.decrypt(encrypted.encode()).decode()
    
    def validate_request(self, request):
        """验证请求合法性"""
        # 检查签名
        # 检查时间戳
        # 检查IP白名单
        pass
```

### 11.2 审计日志
```python
import json
from datetime import datetime

class AuditLogger:
    """审计日志记录器"""
    
    def log_trade(self, trade_data):
        """记录交易日志"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'trade',
            'data': trade_data,
            'user': self.get_current_user(),
            'ip': self.get_client_ip(),
            'signature': self.sign_data(trade_data)
        }
        
        # 写入不可变日志
        self._write_immutable_log(audit_entry)
```

## 12. 持续改进流程

### 12.1 代码审查清单
- [ ] 代码符合命名规范
- [ ] 所有函数有完整的docstring
- [ ] 类型注解完整且正确
- [ ] 异常处理恰当
- [ ] 没有硬编码的配置
- [ ] 性能测试通过
- [ ] 单元测试覆盖率达标
- [ ] 文档已更新
- [ ] 安全检查通过
- [ ] 代码可复用性评估

### 12.2 版本发布规范
```bash
# 版本号规则: MAJOR.MINOR.PATCH
# MAJOR: 不兼容的API变更
# MINOR: 向后兼容的功能新增
# PATCH: 向后兼容的问题修复

# 发布流程
1. 运行完整测试套件
2. 更新CHANGELOG.md
3. 更新版本号
4. 创建git tag
5. 构建Docker镜像
6. 推送到仓库
7. 更新文档
```

## 13. 团队协作规范

### 13.1 Git工作流
```bash
# 分支命名
- main: 生产环境代码
- develop: 开发分支
- feature/xxx: 功能分支
- bugfix/xxx: 缺陷修复
- hotfix/xxx: 紧急修复

# Commit信息格式
<type>(<scope>): <subject>

# type可选值
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 重构
- perf: 性能优化
- test: 测试相关
- chore: 构建/辅助工具变动

# 示例
feat(strategy): add PPO algorithm implementation
fix(data): handle missing values in OHLCV data
docs(api): update trading environment documentation
```

### 13.2 知识共享
- 每周技术分享会
- 维护内部Wiki
- 代码示例库
- 最佳实践文档
- 问题追踪系统

## 14. 参考资源

### 14.1 必读论文
1. "Deep Reinforcement Learning for Trading" (2020)
2. "Practical Deep Reinforcement Learning Approach for Stock Trading" (2018)
3. "Multi-Agent Reinforcement Learning for Liquidation Strategy Analysis" (2019)

### 14.2 推荐项目
- https://github.com/AI4Finance-Foundation/FinRL
- https://github.com/AminHP/gym-anytrading
- https://github.com/tensortrade-org/tensortrade

### 14.3 技术博客
- https://quantdare.com/
- https://www.quantstart.com/
- https://hudsonthames.org/

---

**文档版本**: 1.0.0
**最后更新**: 2024
**维护者**: Claude Code Development Team
