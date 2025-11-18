# 强化学习交易终端 - 模块说明文档

## 📚 文档说明
本文档维护所有模块的详细接口说明、数据结构定义和使用示例，便于问题定位和模块复用。

---

## 🏗️ 系统架构总览

```
rl_trading_system/
├── core/                      # 核心模块
│   ├── __init__.py
│   ├── base.py               # 基类定义
│   ├── config.py             # 配置管理
│   └── exceptions.py         # 异常定义
├── data/                      # 数据处理
│   ├── __init__.py
│   ├── collector.py          # 数据采集
│   ├── preprocessor.py      # 数据预处理
│   ├── feature_engineer.py  # 特征工程
│   └── storage.py           # 数据存储
├── environment/              # 交易环境
│   ├── __init__.py
│   ├── trading_env.py      # 主环境类
│   ├── market_sim.py       # 市场模拟
│   └── reward.py           # 奖励函数
├── agents/                   # RL智能体
│   ├── __init__.py
│   ├── dqn.py              # DQN算法
│   ├── ppo.py              # PPO算法
│   ├── sac.py              # SAC算法
│   └── base_agent.py       # 智能体基类
├── strategies/              # 交易策略
│   ├── __init__.py
│   ├── signals.py          # 信号生成
│   ├── portfolio.py        # 组合管理
│   └── execution.py        # 执行逻辑
├── risk/                    # 风险管理
│   ├── __init__.py
│   ├── metrics.py          # 风险指标
│   ├── controls.py         # 风险控制
│   └── monitor.py          # 风险监控
├── backtest/               # 回测系统
│   ├── __init__.py
│   ├── engine.py          # 回测引擎
│   ├── analyzer.py        # 性能分析
│   └── report.py          # 报告生成
└── utils/                  # 工具函数
    ├── __init__.py
    ├── logger.py          # 日志工具
    ├── metrics.py         # 监控指标
    └── validators.py      # 验证工具
```

---

## 📦 核心模块 (core)

### 🔷 BaseModule
**文件**: `core/base.py`
**描述**: 所有模块的抽象基类

#### 类定义
```python
class BaseModule(ABC):
    def __init__(self, config: Dict[str, Any])
    def validate_config(self) -> bool
    def initialize(self) -> None
    def process(self, *args, **kwargs) -> Any
    def cleanup(self) -> None
```

#### 输入/输出
| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__` | `config: Dict` | `None` | 初始化配置 |
| `validate_config` | `None` | `bool` | 验证配置有效性 |
| `process` | `*args, **kwargs` | `Any` | 主处理逻辑 |

#### 使用示例
```python
class MyModule(BaseModule):
    def process(self, data):
        # 实现具体逻辑
        return processed_data

module = MyModule(config={'param1': value1})
result = module.process(input_data)
```

### 🔷 Config
**文件**: `core/config.py`
**描述**: 全局配置管理器

#### 数据结构
```python
@dataclass
class TradingConfig:
    # 数据配置
    data_source: str = "binance"
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "1h"
    
    # 模型配置
    model_type: str = "dqn"
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # 风险配置
    max_position_size: float = 0.1
    stop_loss: float = 0.05
    take_profit: float = 0.15
```

---

## 📊 数据处理模块 (data)

### 🔷 DataCollector
**文件**: `data/collector.py`
**描述**: 多源数据采集器

#### 接口定义
```python
class DataCollector:
    def fetch_ohlcv(
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame
    
    def stream_realtime(
        symbol: str,
        callback: Callable
    ) -> None
    
    def fetch_orderbook(
        symbol: str,
        limit: int = 20
    ) -> Dict[str, List[Tuple[float, float]]]
```

#### 返回数据结构
```python
# OHLCV DataFrame
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
dtypes = {
    'timestamp': 'datetime64[ns]',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64'
}

# OrderBook Structure
orderbook = {
    'bids': [(price1, quantity1), (price2, quantity2), ...],
    'asks': [(price1, quantity1), (price2, quantity2), ...],
    'timestamp': datetime.utcnow()
}
```

### 🔷 FeatureEngineer
**文件**: `data/feature_engineer.py`
**描述**: 特征工程处理器

#### 主要方法
```python
class FeatureEngineer:
    def add_technical_indicators(
        data: pd.DataFrame,
        indicators: List[str]
    ) -> pd.DataFrame
    
    def create_lag_features(
        data: pd.DataFrame,
        columns: List[str],
        lags: List[int]
    ) -> pd.DataFrame
    
    def add_microstructure_features(
        data: pd.DataFrame,
        orderbook: Dict
    ) -> pd.DataFrame
```

#### 支持的技术指标
| 指标类别 | 指标名称 | 参数 | 输出列 |
|---------|---------|------|--------|
| 趋势 | SMA | period=20 | `sma_20` |
| 趋势 | EMA | period=12 | `ema_12` |
| 动量 | RSI | period=14 | `rsi_14` |
| 动量 | MACD | fast=12, slow=26, signal=9 | `macd`, `macd_signal`, `macd_hist` |
| 波动 | BB | period=20, std=2 | `bb_upper`, `bb_middle`, `bb_lower` |
| 成交量 | VWAP | - | `vwap` |
| 成交量 | OBV | - | `obv` |

#### 特征处理示例
```python
# 初始化
fe = FeatureEngineer()

# 添加技术指标
data = fe.add_technical_indicators(
    data=ohlcv_df,
    indicators=['RSI', 'MACD', 'BB']
)

# 创建滞后特征
data = fe.create_lag_features(
    data=data,
    columns=['close', 'volume'],
    lags=[1, 2, 3, 5, 10]
)

# 添加微观结构特征
data = fe.add_microstructure_features(
    data=data,
    orderbook=orderbook_data
)
```

---

## 🎮 交易环境模块 (environment)

### 🔷 TradingEnvironment
**文件**: `environment/trading_env.py`
**描述**: OpenAI Gym兼容的交易环境

#### 环境规格
```python
class TradingEnvironment(gym.Env):
    # 观察空间
    observation_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(window_size, n_features),
        dtype=np.float32
    )
    
    # 动作空间
    action_space = Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
    # 或
    action_space = Box(
        low=-1,  # 最大卖出
        high=1,  # 最大买入
        shape=(n_assets,),
        dtype=np.float32
    )
```

#### 核心方法
```python
def reset(self) -> np.ndarray:
    """重置环境
    Returns:
        observation: 初始观察值
    """
    
def step(self, action: Union[int, np.ndarray]) -> Tuple:
    """执行动作
    Args:
        action: 交易动作
    Returns:
        observation: 新的观察值
        reward: 奖励值
        done: 是否结束
        info: 额外信息
    """
    
def render(self, mode: str = 'human') -> None:
    """可视化环境状态"""
```

#### 状态表示
```python
# 观察值结构
observation = {
    'market_features': np.array([...]),  # shape: (window, features)
    'account_state': {
        'balance': float,
        'positions': Dict[str, float],
        'total_value': float
    },
    'order_book': {
        'bid_ask_spread': float,
        'depth_imbalance': float
    }
}
```

### 🔷 RewardFunction
**文件**: `environment/reward.py`
**描述**: 奖励函数集合

#### 可用奖励函数
```python
class RewardFunctions:
    @staticmethod
    def simple_pnl(prev_value: float, curr_value: float) -> float:
        """简单盈亏奖励"""
        return curr_value - prev_value
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free: float = 0) -> float:
        """夏普比率奖励"""
        excess_returns = returns - risk_free
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
    
    @staticmethod
    def risk_adjusted(
        pnl: float,
        max_drawdown: float,
        trade_cost: float
    ) -> float:
        """风险调整奖励"""
        return pnl - 0.5 * max_drawdown - trade_cost
```

---

## 🤖 智能体模块 (agents)

### 🔷 DQNAgent
**文件**: `agents/dqn.py`
**描述**: Deep Q-Network智能体

#### 初始化参数
```python
DQNAgent(
    state_dim: int,           # 状态维度
    action_dim: int,          # 动作维度
    learning_rate: float,     # 学习率
    gamma: float,            # 折扣因子
    epsilon: float,          # 探索率
    epsilon_decay: float,    # 探索衰减
    epsilon_min: float,      # 最小探索率
    memory_size: int,        # 经验回放大小
    batch_size: int,         # 批大小
    target_update: int       # 目标网络更新频率
)
```

#### 核心方法
```python
def act(self, state: np.ndarray) -> int:
    """选择动作"""
    
def remember(
    self,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool
) -> None:
    """存储经验"""
    
def replay(self) -> float:
    """经验回放训练
    Returns:
        loss: 训练损失
    """
    
def save(self, path: str) -> None:
    """保存模型"""
    
def load(self, path: str) -> None:
    """加载模型"""
```

### 🔷 PPOAgent
**文件**: `agents/ppo.py`
**描述**: Proximal Policy Optimization智能体

#### 网络架构
```python
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
```

---

## 📈 策略模块 (strategies)

### 🔷 SignalGenerator
**文件**: `strategies/signals.py`
**描述**: 交易信号生成器

#### 信号结构
```python
@dataclass
class TradingSignal:
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 - 1.0
    quantity: float
    price: float
    confidence: float
    metadata: Dict[str, Any]
```

#### 信号生成方法
```python
class SignalGenerator:
    def generate_from_model(
        self,
        state: np.ndarray,
        model: Any
    ) -> TradingSignal:
        """从模型预测生成信号"""
        
    def generate_from_rules(
        self,
        data: pd.DataFrame,
        rules: List[Callable]
    ) -> List[TradingSignal]:
        """从规则生成信号"""
        
    def combine_signals(
        self,
        signals: List[TradingSignal],
        weights: List[float]
    ) -> TradingSignal:
        """组合多个信号"""
```

### 🔷 PortfolioManager
**文件**: `strategies/portfolio.py`
**描述**: 投资组合管理器

#### 持仓结构
```python
@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    pnl: float
    pnl_percent: float
    
class Portfolio:
    cash: float
    positions: Dict[str, Position]
    total_value: float
    leverage: float
    margin_used: float
```

#### 管理方法
```python
class PortfolioManager:
    def calculate_position_size(
        self,
        capital: float,
        risk_per_trade: float,
        stop_loss: float
    ) -> float:
        """Kelly准则计算仓位"""
        
    def rebalance(
        self,
        target_weights: Dict[str, float]
    ) -> List[Order]:
        """重新平衡组合"""
        
    def get_portfolio_metrics(self) -> Dict:
        """获取组合指标"""
        return {
            'total_value': float,
            'cash': float,
            'positions_value': float,
            'unrealized_pnl': float,
            'realized_pnl': float,
            'win_rate': float,
            'avg_win': float,
            'avg_loss': float
        }
```

---

## ⚠️ 风险管理模块 (risk)

### 🔷 RiskMetrics
**文件**: `risk/metrics.py`
**描述**: 风险指标计算

#### 指标计算方法
```python
class RiskMetrics:
    @staticmethod
    def calculate_var(
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """计算VaR"""
        
    @staticmethod
    def calculate_cvar(
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """计算CVaR"""
        
    @staticmethod
    def calculate_max_drawdown(
        equity_curve: np.ndarray
    ) -> Tuple[float, int, int]:
        """计算最大回撤
        Returns:
            max_dd: 最大回撤值
            peak_idx: 峰值索引
            trough_idx: 谷值索引
        """
```

#### 风险报告结构
```python
risk_report = {
    'var_95': float,
    'cvar_95': float,
    'max_drawdown': float,
    'sharpe_ratio': float,
    'sortino_ratio': float,
    'calmar_ratio': float,
    'beta': float,
    'correlation_matrix': np.ndarray,
    'stress_test_results': Dict[str, float]
}
```

### 🔷 RiskController
**文件**: `risk/controls.py`
**描述**: 风险控制器

#### 控制规则
```python
class RiskController:
    def check_position_limit(
        self,
        symbol: str,
        quantity: float
    ) -> bool:
        """检查仓位限制"""
        
    def check_drawdown_limit(
        self,
        current_drawdown: float
    ) -> bool:
        """检查回撤限制"""
        
    def apply_stop_loss(
        self,
        position: Position
    ) -> Optional[Order]:
        """应用止损"""
        
    def calculate_risk_budget(
        self,
        total_capital: float
    ) -> Dict[str, float]:
        """计算风险预算分配"""
```

---

## 🔄 回测模块 (backtest)

### 🔷 BacktestEngine
**文件**: `backtest/engine.py`
**描述**: 事件驱动回测引擎

#### 回测配置
```python
@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission: float = 0.001
    slippage: float = 0.0005
    benchmark: str = "SPY"
    frequency: str = "daily"
```

#### 回测流程
```python
class BacktestEngine:
    def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        config: BacktestConfig
    ) -> BacktestResult:
        """运行回测"""
        
    def walk_forward_analysis(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        train_period: int,
        test_period: int
    ) -> List[BacktestResult]:
        """滚动窗口分析"""
```

#### 回测结果
```python
@dataclass
class BacktestResult:
    # 收益指标
    total_return: float
    annual_return: float
    sharpe_ratio: float
    
    # 风险指标
    max_drawdown: float
    var_95: float
    
    # 交易统计
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_return: float
    
    # 时间序列
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
```

---

## 🛠️ 工具模块 (utils)

### 🔷 Logger
**文件**: `utils/logger.py`
**描述**: 统一日志管理

#### 日志配置
```python
# 日志级别
DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
CRITICAL = 50

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 使用示例
logger = get_logger(__name__)
logger.info("Starting backtest")
logger.error("Failed to execute order", exc_info=True)
```

### 🔷 Validators
**文件**: `utils/validators.py`
**描述**: 数据验证工具

#### 验证函数
```python
def validate_ohlcv(data: pd.DataFrame) -> bool:
    """验证OHLCV数据完整性"""
    
def validate_order(order: Order) -> bool:
    """验证订单合法性"""
    
def validate_config(config: Dict) -> bool:
    """验证配置参数"""
```

---

## 🔌 集成示例

### 完整交易流程
```python
# 1. 初始化配置
config = TradingConfig(
    symbols=['BTC/USDT', 'ETH/USDT'],
    model_type='dqn',
    max_position_size=0.1
)

# 2. 数据准备
collector = DataCollector(config)
data = collector.fetch_ohlcv('BTC/USDT', '1h', start, end)

fe = FeatureEngineer()
features = fe.add_technical_indicators(data, ['RSI', 'MACD'])

# 3. 环境创建
env = TradingEnvironment(
    data=features,
    initial_capital=10000,
    commission=0.001
)

# 4. 智能体训练
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        if len(agent.memory) > batch_size:
            agent.replay()

# 5. 策略执行
signal_gen = SignalGenerator()
portfolio = PortfolioManager(initial_capital=10000)
risk_ctrl = RiskController(max_drawdown=0.2)

for new_data in data_stream:
    # 生成信号
    signal = signal_gen.generate_from_model(new_data, agent)
    
    # 风险检查
    if risk_ctrl.check_position_limit(signal.symbol, signal.quantity):
        # 执行交易
        order = portfolio.create_order(signal)
        execute_order(order)

# 6. 性能分析
analyzer = PerformanceAnalyzer()
results = analyzer.analyze(portfolio.get_history())
print(results.to_report())
```

---

## 🐛 故障排查指南

### 常见问题定位

| 问题 | 可能原因 | 排查模块 | 解决方案 |
|------|---------|---------|---------|
| 数据缺失 | API限制/网络问题 | `data.collector` | 检查API配额，添加重试机制 |
| 特征NaN | 计算错误/数据不足 | `data.feature_engineer` | 增加数据验证，使用fillna |
| 训练不收敛 | 超参数不当 | `agents.*` | 调整学习率，检查奖励函数 |
| 回测偏差大 | 过拟合/数据泄露 | `backtest.engine` | 使用walk-forward，检查look-ahead bias |
| 风险超限 | 仓位过大 | `risk.controller` | 调整position sizing，加强风控 |

### 调试工具
```python
# 开启详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 性能分析
import cProfile
profiler = cProfile.Profile()
profiler.enable()
# ... 代码执行 ...
profiler.disable()
profiler.print_stats()

# 内存分析
from memory_profiler import profile
@profile
def memory_intensive_function():
    pass
```

---

## 📝 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0.0 | 2024-01 | 初始版本，基础模块实现 |
| v1.1.0 | 2024-02 | 添加PPO/SAC算法，优化特征工程 |
| v1.2.0 | 2024-03 | 集成实盘交易接口，增强风控 |

---

## 📚 参考文献

1. **强化学习交易**
   - "Deep Reinforcement Learning for Automated Stock Trading" (2020)
   - GitHub: [FinRL](https://github.com/AI4Finance-Foundation/FinRL)

2. **特征工程**
   - "Advances in Financial Machine Learning" - Marcos Lopez de Prado
   - GitHub: [mlfinlab](https://github.com/hudson-and-thames/mlfinlab)

3. **风险管理**
   - "Quantitative Risk Management" - McNeil, Frey, Embrechts
   - QuantLib Documentation

---

**维护说明**: 本文档随代码更新同步维护，每次添加新模块或修改接口时必须更新相应章节。
