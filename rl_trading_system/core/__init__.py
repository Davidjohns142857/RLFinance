"""
RL Trading System - Core Module

This module provides the foundational components for the trading system,
including base classes, configuration management, and exception handling.

Modules:
    - base: Abstract base classes for all system components
    - config: Configuration management and validation
    - exceptions: Custom exception hierarchy

Example:
    >>> from rl_trading_system.core import TradingConfig, BaseModule
    >>> config = TradingConfig()
    >>> print(config.model.learning_rate)
    0.001
"""

from rl_trading_system.core.base import (
    BaseModule,
    BaseStrategy,
    BaseAgent
)

from rl_trading_system.core.config import (
    TradingConfig,
    DataConfig,
    ModelConfig,
    RiskConfig,
    EnvironmentConfig,
    BacktestConfig,
    LoggingConfig,
    ModelType,
    DataSource,
    Timeframe
)

from rl_trading_system.core.exceptions import (
    TradingException,
    ConfigException,
    DataException,
    DataCollectionException,
    DataValidationException,
    FeatureEngineeringException,
    ExecutionException,
    OrderException,
    PositionException,
    RiskException,
    DrawdownException,
    PositionLimitException,
    ModelException,
    TrainingException,
    InferenceException,
    EnvironmentException,
    BacktestException,
    StrategyException,
    StorageException,
    NetworkException,
    ValidationException
)

__version__ = "1.0.0"

__all__ = [
    # Base classes
    'BaseModule',
    'BaseStrategy',
    'BaseAgent',

    # Configuration
    'TradingConfig',
    'DataConfig',
    'ModelConfig',
    'RiskConfig',
    'EnvironmentConfig',
    'BacktestConfig',
    'LoggingConfig',
    'ModelType',
    'DataSource',
    'Timeframe',

    # Exceptions
    'TradingException',
    'ConfigException',
    'DataException',
    'DataCollectionException',
    'DataValidationException',
    'FeatureEngineeringException',
    'ExecutionException',
    'OrderException',
    'PositionException',
    'RiskException',
    'DrawdownException',
    'PositionLimitException',
    'ModelException',
    'TrainingException',
    'InferenceException',
    'EnvironmentException',
    'BacktestException',
    'StrategyException',
    'StorageException',
    'NetworkException',
    'ValidationException',
]
