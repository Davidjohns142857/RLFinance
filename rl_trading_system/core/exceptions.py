"""
Module: core.exceptions
Description: Custom exceptions for the RL trading system
Author: Claude Code
Date: 2025-11-18
Version: 1.0.0

This module defines a hierarchy of custom exceptions used throughout
the trading system for better error handling and debugging.

Usage Example:
    >>> from rl_trading_system.core.exceptions import DataException
    >>> raise DataException("Invalid OHLCV data")
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TradingException(Exception):
    """
    Base exception class for all trading system exceptions

    All custom exceptions should inherit from this class to enable
    consistent exception handling across the system.

    Attributes:
        message (str): Error message
        timestamp (datetime): When the exception occurred
        context (Dict): Additional context information
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trading exception

        Args:
            message: Descriptive error message
            context: Additional context information (optional)
        """
        super().__init__(message)
        self.message = message
        self.timestamp = datetime.utcnow()
        self.context = context or {}
        self._log_exception()

    def _log_exception(self) -> None:
        """Log the exception with context"""
        logger.error(
            f"{self.__class__.__name__}: {self.message}",
            extra={'context': self.context, 'timestamp': self.timestamp}
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for serialization

        Returns:
            Dictionary representation of the exception
        """
        return {
            'exception_type': self.__class__.__name__,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }


class ConfigException(TradingException):
    """
    Raised when configuration is invalid or missing

    Example:
        >>> raise ConfigException(
        ...     "Missing API key",
        ...     context={'key_name': 'binance_api_key'}
        ... )
    """
    pass


class DataException(TradingException):
    """
    Raised for data-related errors

    This includes data collection, validation, preprocessing, and storage errors.

    Example:
        >>> raise DataException(
        ...     "Invalid OHLCV data",
        ...     context={'symbol': 'BTC/USDT', 'reason': 'missing columns'}
        ... )
    """
    pass


class DataCollectionException(DataException):
    """
    Raised when data collection fails

    Example:
        >>> raise DataCollectionException(
        ...     "API request failed",
        ...     context={'exchange': 'binance', 'error_code': 429}
        ... )
    """
    pass


class DataValidationException(DataException):
    """
    Raised when data validation fails

    Example:
        >>> raise DataValidationException(
        ...     "Data quality check failed",
        ...     context={'check': 'missing_values', 'threshold': 0.05}
        ... )
    """
    pass


class FeatureEngineeringException(DataException):
    """
    Raised when feature engineering fails

    Example:
        >>> raise FeatureEngineeringException(
        ...     "Failed to calculate RSI",
        ...     context={'indicator': 'RSI', 'period': 14}
        ... )
    """
    pass


class ExecutionException(TradingException):
    """
    Raised for trading execution errors

    This includes order placement, modification, and cancellation errors.

    Attributes:
        order_id (Optional[str]): Associated order ID if applicable
    """

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize execution exception

        Args:
            message: Error message
            order_id: Order ID if applicable
            context: Additional context
        """
        context = context or {}
        if order_id:
            context['order_id'] = order_id
        super().__init__(message, context)
        self.order_id = order_id


class OrderException(ExecutionException):
    """
    Raised when order operations fail

    Example:
        >>> raise OrderException(
        ...     "Order placement failed",
        ...     order_id="12345",
        ...     context={'symbol': 'BTC/USDT', 'reason': 'insufficient balance'}
        ... )
    """
    pass


class PositionException(ExecutionException):
    """
    Raised when position operations fail

    Example:
        >>> raise PositionException(
        ...     "Failed to close position",
        ...     context={'symbol': 'ETH/USDT', 'quantity': 1.5}
        ... )
    """
    pass


class RiskException(TradingException):
    """
    Raised for risk management violations

    This is a critical exception that should trigger immediate attention.

    Attributes:
        risk_type (str): Type of risk violation
        severity (str): Severity level (low, medium, high, critical)
        position (Optional[Dict]): Associated position if applicable
        risk_metrics (Optional[Dict]): Risk metrics at time of violation
    """

    def __init__(
        self,
        message: str,
        risk_type: str,
        severity: str = 'high',
        position: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize risk exception

        Args:
            message: Error message
            risk_type: Type of risk (e.g., 'max_drawdown', 'position_limit')
            severity: Severity level
            position: Position information
            risk_metrics: Risk metrics
            context: Additional context
        """
        context = context or {}
        context.update({
            'risk_type': risk_type,
            'severity': severity
        })
        super().__init__(message, context)
        self.risk_type = risk_type
        self.severity = severity
        self.position = position
        self.risk_metrics = risk_metrics
        self._log_risk_event()

    def _log_risk_event(self) -> None:
        """Log risk event to audit trail"""
        logger.critical(
            f"RISK VIOLATION: {self.risk_type} - {self.message}",
            extra={
                'severity': self.severity,
                'position': self.position,
                'risk_metrics': self.risk_metrics,
                'timestamp': self.timestamp
            }
        )


class DrawdownException(RiskException):
    """
    Raised when drawdown limits are exceeded

    Example:
        >>> raise DrawdownException(
        ...     "Maximum drawdown exceeded",
        ...     risk_type='max_drawdown',
        ...     risk_metrics={'current_dd': 0.25, 'limit': 0.20}
        ... )
    """

    def __init__(
        self,
        message: str,
        current_drawdown: float,
        max_drawdown: float,
        **kwargs
    ):
        """
        Initialize drawdown exception

        Args:
            message: Error message
            current_drawdown: Current drawdown value
            max_drawdown: Maximum allowed drawdown
            **kwargs: Additional arguments
        """
        risk_metrics = kwargs.pop('risk_metrics', {})
        risk_metrics.update({
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown
        })
        super().__init__(
            message,
            risk_type='max_drawdown',
            severity='critical',
            risk_metrics=risk_metrics,
            **kwargs
        )


class PositionLimitException(RiskException):
    """
    Raised when position limits are exceeded

    Example:
        >>> raise PositionLimitException(
        ...     "Position size limit exceeded",
        ...     risk_type='position_limit',
        ...     position={'symbol': 'BTC/USDT', 'size': 2.5},
        ...     risk_metrics={'current': 2.5, 'limit': 2.0}
        ... )
    """
    pass


class ModelException(TradingException):
    """
    Raised for model-related errors

    This includes training, inference, and model management errors.

    Example:
        >>> raise ModelException(
        ...     "Model training failed",
        ...     context={'model_type': 'DQN', 'epoch': 42}
        ... )
    """
    pass


class TrainingException(ModelException):
    """
    Raised when model training fails

    Example:
        >>> raise TrainingException(
        ...     "Training diverged",
        ...     context={'loss': float('inf'), 'episode': 150}
        ... )
    """
    pass


class InferenceException(ModelException):
    """
    Raised when model inference fails

    Example:
        >>> raise InferenceException(
        ...     "Invalid input shape",
        ...     context={'expected': (32, 10), 'received': (32, 8)}
        ... )
    """
    pass


class EnvironmentException(TradingException):
    """
    Raised for trading environment errors

    Example:
        >>> raise EnvironmentException(
        ...     "Environment reset failed",
        ...     context={'env_name': 'TradingEnv-v0'}
        ... )
    """
    pass


class BacktestException(TradingException):
    """
    Raised for backtesting errors

    Example:
        >>> raise BacktestException(
        ...     "Insufficient data for backtest",
        ...     context={'required_days': 365, 'available_days': 180}
        ... )
    """
    pass


class StrategyException(TradingException):
    """
    Raised for strategy-related errors

    Example:
        >>> raise StrategyException(
        ...     "Strategy initialization failed",
        ...     context={'strategy_name': 'MomentumStrategy'}
        ... )
    """
    pass


class StorageException(TradingException):
    """
    Raised for data storage errors

    Example:
        >>> raise StorageException(
        ...     "Database connection failed",
        ...     context={'database': 'influxdb', 'host': 'localhost'}
        ... )
    """
    pass


class NetworkException(TradingException):
    """
    Raised for network-related errors

    Example:
        >>> raise NetworkException(
        ...     "API request timeout",
        ...     context={'endpoint': '/api/v3/ticker', 'timeout': 30}
        ... )
    """
    pass


class ValidationException(TradingException):
    """
    Raised when validation fails

    Example:
        >>> raise ValidationException(
        ...     "Invalid parameter value",
        ...     context={'parameter': 'learning_rate', 'value': -0.1}
        ... )
    """
    pass
