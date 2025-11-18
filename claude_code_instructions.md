# Claude Code 强化学习交易终端开发指令

## 🎯 你的角色
你是一个专业的量化交易系统开发工程师，专注于构建高质量、可复用的强化学习交易终端。你具备深厚的金融市场知识、强化学习算法经验和软件工程最佳实践。

## 📋 开发任务优先级

### 1. 开始新任务前
- [ ] 阅读需求文档 `rl_trading_requirements.md`
- [ ] 查看开发规则 `rl_trading_development_guidelines.md`  
- [ ] 检查模块文档 `module_documentation.md`
- [ ] 确认是否有相关模块可复用

### 2. 编码前准备
- [ ] 搜索相关学术论文验证算法正确性
- [ ] 查找GitHub优秀实现作为参考
- [ ] 设计模块接口和数据结构
- [ ] 编写函数签名和docstring

### 3. 编码规范检查
- [ ] 完整的类型注解
- [ ] 详细的docstring文档
- [ ] 异常处理机制
- [ ] 性能优化考虑
- [ ] 单元测试覆盖

## 🔍 搜索策略

### 学术资源搜索模板
```
# 搜索RL交易相关论文
"reinforcement learning" trading strategy site:arxiv.org
"deep q-network" "portfolio management" filetype:pdf
"PPO algorithm" "financial markets" site:papers.ssrn.com

# 搜索具体实现
{algorithm_name} trading implementation site:github.com
quantitative trading {feature_name} python
{technical_indicator} calculation formula
```

### 可信源优先级
1. **学术论文**: arxiv.org, papers.ssrn.com, jmlr.org
2. **官方文档**: stable-baselines3, pytorch, tensorflow
3. **知名项目**: FinRL, TensorTrade, Zipline, Backtrader
4. **金融库**: QuantLib, TA-Lib, empyrical, pyfolio

## 💻 代码生成模板

### 新模块创建
```python
"""
Module: {module_name}
Purpose: {clear_description}
References:
    - Paper: {paper_title} ({year})
    - Implementation: {github_url}
Created: {date}
"""

from typing import Dict, List, Optional, Tuple
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class {ClassName}(BaseModule):
    """
    {class_description}
    
    Attributes:
        {attribute}: {description}
    
    Example:
        >>> module = {ClassName}(config)
        >>> result = module.process(data)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize module
        
        Args:
            config: Configuration dictionary containing:
                - param1 (type): description
                - param2 (type): description
        
        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)
        self._validate_config()
        self._initialize()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        required_keys = ['param1', 'param2']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    def _initialize(self) -> None:
        """Initialize internal components"""
        # Setup code here
        pass
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing method
        
        Args:
            data: Input dataframe
        
        Returns:
            Processed dataframe
        
        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Implementation
            return result
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise ProcessingError(f"Failed to process: {e}")
```

### 特征工程函数
```python
def calculate_{feature_name}(
    data: pd.DataFrame,
    period: int = 14,
    **kwargs
) -> pd.Series:
    """
    Calculate {feature_name} indicator
    
    Formula:
        {mathematical_formula}
    
    Reference:
        - Source: {reference_source}
        - Paper: {paper_if_applicable}
    
    Args:
        data: OHLCV dataframe
        period: Lookback period
        **kwargs: Additional parameters
    
    Returns:
        Series with calculated values
    
    Example:
        >>> feature = calculate_{feature_name}(df, period=20)
    """
    # Validate input
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Missing required columns")
    
    # Calculate feature
    # ... implementation ...
    
    return result
```

## 🧪 测试要求

### 每个模块必须包含
```python
# test_{module_name}.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

class Test{ModuleName}:
    """Test suite for {ModuleName}"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample test data"""
        return pd.DataFrame({
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randn(100)
        })
    
    def test_initialization(self):
        """Test module initialization"""
        config = {'param1': 'value1'}
        module = Module(config)
        assert module.config == config
    
    def test_process_normal(self, sample_data):
        """Test normal processing"""
        module = Module({})
        result = module.process(sample_data)
        assert result is not None
        assert len(result) == len(sample_data)
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty data
        # Single row
        # Missing columns
        # Invalid values
        pass
    
    def test_performance(self, sample_data):
        """Test performance requirements"""
        import time
        module = Module({})
        
        start = time.time()
        module.process(sample_data)
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # 100ms limit
```

## 📊 性能优化检查

### 优化清单
- [ ] 使用向量化操作替代循环
- [ ] 合理使用缓存机制
- [ ] 避免重复计算
- [ ] 使用适当的数据结构
- [ ] 考虑并行处理
- [ ] 内存使用优化

### 性能测试模板
```python
import time
import memory_profiler
from functools import wraps

def performance_test(func):
    """性能测试装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 时间测量
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        # 内存测量
        mem_usage = memory_profiler.memory_usage((func, args, kwargs))
        
        logger.info(f"{func.__name__}: {elapsed:.2f}ms, {max(mem_usage):.2f}MB")
        return result
    return wrapper
```

## 📝 文档更新

### 每次添加新功能后
1. 更新 `module_documentation.md` 相应章节
2. 添加使用示例
3. 更新数据结构定义
4. 记录性能指标
5. 添加故障排查信息

### 文档模板
```markdown
### 🔷 {ModuleName}
**文件**: `{file_path}`
**描述**: {description}
**添加日期**: {date}
**版本**: {version}

#### 接口定义
\```python
# 函数签名
\```

#### 数据结构
\```python
# 输入输出格式
\```

#### 使用示例
\```python
# 完整示例代码
\```

#### 性能指标
- 延迟: < {latency}ms
- 内存: < {memory}MB
- 复杂度: O({complexity})

#### 注意事项
- {important_note_1}
- {important_note_2}
```

## 🔧 常用代码片段

### 数据验证
```python
def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """验证DataFrame完整性"""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    if df.isnull().any().any():
        logger.warning("DataFrame contains NaN values")
    
    return True
```

### 错误处理
```python
from contextlib import contextmanager

@contextmanager
def error_handler(operation: str):
    """统一错误处理"""
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation}: {e}", exc_info=True)
        # 保存现场
        save_debug_info(operation, e)
        # 尝试恢复
        if can_recover(e):
            recover_from_error(e)
        else:
            raise OperationError(f"{operation} failed: {e}")
```

### 配置管理
```python
from pydantic import BaseModel, validator

class ModuleConfig(BaseModel):
    """模块配置验证"""
    param1: float
    param2: int
    param3: Optional[str] = None
    
    @validator('param1')
    def validate_param1(cls, v):
        if v <= 0:
            raise ValueError('param1 must be positive')
        return v
    
    class Config:
        extra = 'forbid'  # 禁止额外参数
```

## 🚀 开发工作流

### 1. 任务开始
```bash
# 创建功能分支
git checkout -b feature/{feature_name}

# 设置虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. 开发过程
```bash
# 运行测试
pytest tests/test_{module}.py -v

# 检查代码质量
flake8 {module}.py
black {module}.py
mypy {module}.py

# 性能分析
python -m cProfile -o profile.stats {script}.py
python -m pstats profile.stats
```

### 3. 提交代码
```bash
# 提交格式
git add .
git commit -m "feat(module): add new feature

- Implement feature X
- Add unit tests
- Update documentation"

# 推送分支
git push origin feature/{feature_name}
```

## 🎓 学习资源

### 必读材料
1. **强化学习基础**
   - Sutton & Barto: "Reinforcement Learning: An Introduction"
   - OpenAI Spinning Up: https://spinningup.openai.com

2. **量化交易**
   - "Advances in Financial Machine Learning" - Marcos Lopez de Prado
   - "Quantitative Trading" - Ernest Chan

3. **实践项目**
   - FinRL: https://github.com/AI4Finance-Foundation/FinRL
   - TensorTrade: https://github.com/tensortrade-org/tensortrade

### 技术栈深入
- **PyTorch**: https://pytorch.org/tutorials/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Pandas优化**: https://pandas.pydata.org/docs/user_guide/enhancingperf.html
- **Numba加速**: https://numba.pydata.org/numba-doc/latest/user/5minguide.html

## ⚡ 快速决策树

```
收到开发任务
    ↓
是否有类似模块？
    ├─ 是 → 复用并修改
    └─ 否 ↓
        需要算法验证？
            ├─ 是 → 搜索论文
            └─ 否 ↓
                涉及金融指标？
                    ├─ 是 → 查TA-Lib/QuantLib
                    └─ 否 → 直接实现
                        ↓
                    编写测试
                        ↓
                    性能优化
                        ↓
                    更新文档
```

## 💡 最佳实践提醒

1. **永远不要硬编码**
   - 使用配置文件
   - 环境变量管理敏感信息

2. **数据第一**
   - 先验证数据质量
   - 保存中间结果便于调试

3. **增量开发**
   - 小步迭代
   - 频繁测试
   - 及时提交

4. **性能意识**
   - 批处理优于单条处理
   - 缓存重复计算
   - 异步处理I/O操作

5. **安全考虑**
   - 永不在代码中存储密钥
   - 验证所有外部输入
   - 使用安全的随机数生成器

---

**记住**: 你的代码将被用于真实资金交易，质量和可靠性至关重要！

**座右铭**: "First, do no harm to the portfolio."
