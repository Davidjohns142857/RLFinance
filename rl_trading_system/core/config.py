"""
Module: core.config
Description: Configuration management for the RL trading system
Author: Claude Code
Date: 2025-11-18
Version: 1.0.0

Dependencies:
    - pydantic: Configuration validation
    - os: Environment variables
    - pathlib: Path handling

Usage Example:
    >>> from rl_trading_system.core.config import TradingConfig
    >>> config = TradingConfig(
    ...     symbols=['BTC/USDT'],
    ...     model_type='dqn'
    ... )
    >>> print(config.learning_rate)
    0.001
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Supported reinforcement learning model types"""
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"
    PPO = "ppo"
    SAC = "sac"
    A3C = "a3c"
    TD3 = "td3"


class DataSource(str, Enum):
    """Supported data sources"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    YAHOO = "yahoo"
    ALPACA = "alpaca"
    CUSTOM = "custom"


class Timeframe(str, Enum):
    """Supported timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


@dataclass
class DataConfig:
    """
    Data configuration

    Attributes:
        source: Data source
        symbols: List of trading symbols
        timeframe: Candlestick timeframe
        lookback_period: Historical data lookback in days
        cache_enabled: Whether to enable data caching
        cache_ttl: Cache time-to-live in seconds
    """
    source: str = DataSource.BINANCE.value
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT'])
    timeframe: str = Timeframe.H1.value
    lookback_period: int = 365
    cache_enabled: bool = True
    cache_ttl: int = 3600

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.symbols:
            raise ValueError("At least one symbol must be specified")
        if self.lookback_period <= 0:
            raise ValueError("Lookback period must be positive")
        if self.cache_ttl <= 0:
            raise ValueError("Cache TTL must be positive")


@dataclass
class ModelConfig:
    """
    Model configuration

    Attributes:
        model_type: Type of RL model
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        gamma: Discount factor
        epsilon: Exploration rate (for epsilon-greedy)
        epsilon_decay: Epsilon decay rate
        epsilon_min: Minimum epsilon value
        memory_size: Replay memory size
        target_update_freq: Target network update frequency
        hidden_dims: Hidden layer dimensions
    """
    model_type: str = ModelType.DQN.value
    learning_rate: float = 0.001
    batch_size: int = 32
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    memory_size: int = 10000
    target_update_freq: int = 100
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1")
        if not 0 <= self.epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1")
        if not 0 <= self.epsilon_decay <= 1:
            raise ValueError("Epsilon decay must be between 0 and 1")
        if not 0 <= self.epsilon_min <= 1:
            raise ValueError("Epsilon min must be between 0 and 1")


@dataclass
class RiskConfig:
    """
    Risk management configuration

    Attributes:
        max_position_size: Maximum position size as fraction of capital
        max_drawdown: Maximum allowed drawdown
        stop_loss: Stop loss percentage
        take_profit: Take profit percentage
        max_leverage: Maximum leverage allowed
        risk_per_trade: Risk per trade as fraction of capital
        var_confidence: VaR confidence level
    """
    max_position_size: float = 0.1
    max_drawdown: float = 0.2
    stop_loss: float = 0.05
    take_profit: float = 0.15
    max_leverage: float = 1.0
    risk_per_trade: float = 0.02
    var_confidence: float = 0.95

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not 0 < self.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
        if not 0 < self.max_drawdown <= 1:
            raise ValueError("Max drawdown must be between 0 and 1")
        if self.stop_loss <= 0:
            raise ValueError("Stop loss must be positive")
        if self.take_profit <= 0:
            raise ValueError("Take profit must be positive")
        if self.max_leverage < 1:
            raise ValueError("Max leverage must be >= 1")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("Risk per trade must be between 0 and 1")
        if not 0 < self.var_confidence < 1:
            raise ValueError("VaR confidence must be between 0 and 1")


@dataclass
class EnvironmentConfig:
    """
    Trading environment configuration

    Attributes:
        initial_capital: Initial trading capital
        commission: Trading commission rate
        slippage: Slippage rate
        window_size: Observation window size
        max_steps: Maximum steps per episode
        reward_scaling: Reward scaling factor
    """
    initial_capital: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0005
    window_size: int = 20
    max_steps: int = 10000
    reward_scaling: float = 1.0

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.commission < 0:
            raise ValueError("Commission must be non-negative")
        if self.slippage < 0:
            raise ValueError("Slippage must be non-negative")
        if self.window_size <= 0:
            raise ValueError("Window size must be positive")
        if self.max_steps <= 0:
            raise ValueError("Max steps must be positive")


@dataclass
class BacktestConfig:
    """
    Backtesting configuration

    Attributes:
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        walk_forward: Whether to use walk-forward analysis
        train_ratio: Train/test split ratio
        num_folds: Number of folds for cross-validation
    """
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    walk_forward: bool = False
    train_ratio: float = 0.8
    num_folds: int = 5

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not 0 < self.train_ratio < 1:
            raise ValueError("Train ratio must be between 0 and 1")
        if self.num_folds <= 0:
            raise ValueError("Number of folds must be positive")


@dataclass
class LoggingConfig:
    """
    Logging configuration

    Attributes:
        level: Logging level
        log_dir: Directory for log files
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        format: Log message format
    """
    level: str = "INFO"
    log_dir: str = "./logs"
    log_to_file: bool = True
    log_to_console: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class TradingConfig:
    """
    Main trading system configuration

    This is the top-level configuration class that combines all sub-configurations.

    Attributes:
        data: Data configuration
        model: Model configuration
        risk: Risk management configuration
        environment: Environment configuration
        backtest: Backtesting configuration
        logging: Logging configuration

    Example:
        >>> config = TradingConfig()
        >>> config.data.symbols = ['BTC/USDT', 'ETH/USDT']
        >>> config.model.learning_rate = 0.0001
        >>> config.save_to_file('config.json')
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary

        Returns:
            Dictionary representation of configuration
        """
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'risk': asdict(self.risk),
            'environment': asdict(self.environment),
            'backtest': asdict(self.backtest),
            'logging': asdict(self.logging)
        }

    def save_to_file(self, filepath: str) -> None:
        """
        Save configuration to JSON file

        Args:
            filepath: Path to save configuration file

        Example:
            >>> config.save_to_file('config.json')
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'TradingConfig':
        """
        Load configuration from JSON file

        Args:
            filepath: Path to configuration file

        Returns:
            TradingConfig instance

        Example:
            >>> config = TradingConfig.load_from_file('config.json')
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            data=DataConfig(**data.get('data', {})),
            model=ModelConfig(**data.get('model', {})),
            risk=RiskConfig(**data.get('risk', {})),
            environment=EnvironmentConfig(**data.get('environment', {})),
            backtest=BacktestConfig(**data.get('backtest', {})),
            logging=LoggingConfig(**data.get('logging', {}))
        )

    @classmethod
    def load_from_env(cls) -> 'TradingConfig':
        """
        Load configuration from environment variables

        Environment variables should be prefixed with 'TRADING_'

        Returns:
            TradingConfig instance

        Example:
            >>> os.environ['TRADING_MODEL_LEARNING_RATE'] = '0.0001'
            >>> config = TradingConfig.load_from_env()
        """
        config = cls()

        # Load data config from env
        if symbols := os.getenv('TRADING_DATA_SYMBOLS'):
            config.data.symbols = symbols.split(',')
        if source := os.getenv('TRADING_DATA_SOURCE'):
            config.data.source = source
        if timeframe := os.getenv('TRADING_DATA_TIMEFRAME'):
            config.data.timeframe = timeframe

        # Load model config from env
        if lr := os.getenv('TRADING_MODEL_LEARNING_RATE'):
            config.model.learning_rate = float(lr)
        if model_type := os.getenv('TRADING_MODEL_TYPE'):
            config.model.model_type = model_type
        if batch_size := os.getenv('TRADING_MODEL_BATCH_SIZE'):
            config.model.batch_size = int(batch_size)

        # Load risk config from env
        if max_pos := os.getenv('TRADING_RISK_MAX_POSITION_SIZE'):
            config.risk.max_position_size = float(max_pos)
        if max_dd := os.getenv('TRADING_RISK_MAX_DRAWDOWN'):
            config.risk.max_drawdown = float(max_dd)
        if stop_loss := os.getenv('TRADING_RISK_STOP_LOSS'):
            config.risk.stop_loss = float(stop_loss)

        logger.info("Configuration loaded from environment variables")
        return config

    def validate(self) -> bool:
        """
        Validate entire configuration

        Returns:
            True if valid

        Raises:
            ValueError: If any configuration is invalid
        """
        # Validation is performed in __post_init__ of each config class
        logger.info("Configuration validation successful")
        return True

    def __repr__(self) -> str:
        """String representation"""
        return f"TradingConfig(model={self.model.model_type}, symbols={self.data.symbols})"
