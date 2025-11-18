"""
Module: utils.validators
Description: Data validation utilities for the RL trading system
Author: Claude Code
Date: 2025-11-18
Version: 1.0.0

This module provides validation functions for various data types used
throughout the trading system, including market data, orders, and configurations.

Usage Example:
    >>> from rl_trading_system.utils.validators import validate_ohlcv
    >>> is_valid = validate_ohlcv(dataframe)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging

from rl_trading_system.core.exceptions import (
    DataValidationException,
    ValidationException
)

logger = logging.getLogger(__name__)


def validate_ohlcv(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    check_nulls: bool = True,
    check_order: bool = True,
    check_price_validity: bool = True
) -> bool:
    """
    Validate OHLCV (Open, High, Low, Close, Volume) data

    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        check_nulls: Whether to check for null values
        check_order: Whether to check timestamp ordering
        check_price_validity: Whether to validate price relationships

    Returns:
        True if validation passes

    Raises:
        DataValidationException: If validation fails

    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2020-01-01', periods=10),
        ...     'open': np.random.rand(10) * 100,
        ...     'high': np.random.rand(10) * 100,
        ...     'low': np.random.rand(10) * 100,
        ...     'close': np.random.rand(10) * 100,
        ...     'volume': np.random.rand(10) * 1000
        ... })
        >>> validate_ohlcv(df)
        True
    """
    if required_columns is None:
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    # Check if DataFrame is empty
    if data.empty:
        raise DataValidationException(
            "DataFrame is empty",
            context={'validation': 'ohlcv'}
        )

    # Check for required columns
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise DataValidationException(
            f"Missing required columns: {missing_columns}",
            context={'missing': list(missing_columns)}
        )

    # Check for null values
    if check_nulls:
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            null_info = null_counts[null_counts > 0].to_dict()
            raise DataValidationException(
                "Data contains null values",
                context={'null_counts': null_info}
            )

    # Check timestamp ordering
    if check_order and 'timestamp' in data.columns:
        if not data['timestamp'].is_monotonic_increasing:
            raise DataValidationException(
                "Timestamps are not in ascending order",
                context={'validation': 'timestamp_order'}
            )

    # Validate price relationships (high >= low, high >= open, high >= close)
    if check_price_validity:
        price_cols = ['open', 'high', 'low', 'close']
        if all(col in data.columns for col in price_cols):
            # High should be >= all other prices
            if not (data['high'] >= data['low']).all():
                raise DataValidationException(
                    "High price is less than low price in some rows",
                    context={'validation': 'high_low'}
                )
            if not (data['high'] >= data['open']).all():
                raise DataValidationException(
                    "High price is less than open price in some rows",
                    context={'validation': 'high_open'}
                )
            if not (data['high'] >= data['close']).all():
                raise DataValidationException(
                    "High price is less than close price in some rows",
                    context={'validation': 'high_close'}
                )

            # Low should be <= all other prices
            if not (data['low'] <= data['open']).all():
                raise DataValidationException(
                    "Low price is greater than open price in some rows",
                    context={'validation': 'low_open'}
                )
            if not (data['low'] <= data['close']).all():
                raise DataValidationException(
                    "Low price is greater than close price in some rows",
                    context={'validation': 'low_close'}
                )

            # Check for negative prices
            for col in price_cols:
                if (data[col] <= 0).any():
                    raise DataValidationException(
                        f"Negative or zero prices found in {col}",
                        context={'column': col}
                    )

        # Validate volume
        if 'volume' in data.columns:
            if (data['volume'] < 0).any():
                raise DataValidationException(
                    "Negative volume found",
                    context={'column': 'volume'}
                )

    logger.info(f"OHLCV validation passed for {len(data)} rows")
    return True


def validate_features(
    data: pd.DataFrame,
    allow_inf: bool = False,
    allow_nan: bool = False,
    max_nan_ratio: float = 0.1
) -> bool:
    """
    Validate feature DataFrame

    Args:
        data: Feature DataFrame
        allow_inf: Whether to allow infinite values
        allow_nan: Whether to allow NaN values
        max_nan_ratio: Maximum ratio of NaN values allowed

    Returns:
        True if validation passes

    Raises:
        DataValidationException: If validation fails

    Example:
        >>> features = pd.DataFrame(np.random.rand(100, 10))
        >>> validate_features(features)
        True
    """
    if data.empty:
        raise DataValidationException(
            "Feature DataFrame is empty",
            context={'validation': 'features'}
        )

    # Check for infinite values
    if not allow_inf:
        inf_mask = np.isinf(data.select_dtypes(include=[np.number]))
        if inf_mask.any().any():
            inf_cols = inf_mask.any()[inf_mask.any()].index.tolist()
            raise DataValidationException(
                "Infinite values found in features",
                context={'columns': inf_cols}
            )

    # Check for NaN values
    if not allow_nan:
        nan_counts = data.isnull().sum()
        if nan_counts.any():
            nan_ratio = nan_counts / len(data)
            problematic_cols = nan_ratio[nan_ratio > max_nan_ratio].index.tolist()
            if problematic_cols:
                raise DataValidationException(
                    f"NaN ratio exceeds threshold ({max_nan_ratio})",
                    context={
                        'columns': problematic_cols,
                        'ratios': nan_ratio[problematic_cols].to_dict()
                    }
                )

    logger.info(f"Feature validation passed for shape {data.shape}")
    return True


def validate_order(order: Dict[str, Any]) -> bool:
    """
    Validate order dictionary

    Args:
        order: Order dictionary

    Returns:
        True if validation passes

    Raises:
        ValidationException: If validation fails

    Example:
        >>> order = {
        ...     'symbol': 'BTC/USDT',
        ...     'type': 'market',
        ...     'side': 'buy',
        ...     'quantity': 1.0
        ... }
        >>> validate_order(order)
        True
    """
    required_fields = ['symbol', 'type', 'side', 'quantity']

    # Check required fields
    missing_fields = [f for f in required_fields if f not in order]
    if missing_fields:
        raise ValidationException(
            f"Missing required order fields: {missing_fields}",
            context={'order': order}
        )

    # Validate order type
    valid_types = ['market', 'limit', 'stop', 'stop_limit']
    if order['type'] not in valid_types:
        raise ValidationException(
            f"Invalid order type: {order['type']}",
            context={'valid_types': valid_types}
        )

    # Validate side
    valid_sides = ['buy', 'sell']
    if order['side'] not in valid_sides:
        raise ValidationException(
            f"Invalid order side: {order['side']}",
            context={'valid_sides': valid_sides}
        )

    # Validate quantity
    if order['quantity'] <= 0:
        raise ValidationException(
            "Order quantity must be positive",
            context={'quantity': order['quantity']}
        )

    # Validate price for limit orders
    if order['type'] in ['limit', 'stop_limit']:
        if 'price' not in order or order['price'] <= 0:
            raise ValidationException(
                "Limit orders require a positive price",
                context={'order': order}
            )

    logger.debug(f"Order validation passed: {order['symbol']} {order['side']} {order['quantity']}")
    return True


def validate_position(position: Dict[str, Any]) -> bool:
    """
    Validate position dictionary

    Args:
        position: Position dictionary

    Returns:
        True if validation passes

    Raises:
        ValidationException: If validation fails

    Example:
        >>> position = {
        ...     'symbol': 'BTC/USDT',
        ...     'quantity': 1.5,
        ...     'entry_price': 50000.0,
        ...     'current_price': 51000.0
        ... }
        >>> validate_position(position)
        True
    """
    required_fields = ['symbol', 'quantity', 'entry_price']

    # Check required fields
    missing_fields = [f for f in required_fields if f not in position]
    if missing_fields:
        raise ValidationException(
            f"Missing required position fields: {missing_fields}",
            context={'position': position}
        )

    # Validate quantity (can be negative for short positions)
    if position['quantity'] == 0:
        raise ValidationException(
            "Position quantity cannot be zero",
            context={'position': position}
        )

    # Validate entry price
    if position['entry_price'] <= 0:
        raise ValidationException(
            "Entry price must be positive",
            context={'entry_price': position['entry_price']}
        )

    # Validate current price if present
    if 'current_price' in position and position['current_price'] <= 0:
        raise ValidationException(
            "Current price must be positive",
            context={'current_price': position['current_price']}
        )

    logger.debug(f"Position validation passed: {position['symbol']} {position['quantity']}")
    return True


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate configuration against schema

    Args:
        config: Configuration dictionary
        schema: Schema dictionary with validation rules

    Returns:
        True if validation passes

    Raises:
        ValidationException: If validation fails

    Example:
        >>> schema = {
        ...     'learning_rate': {'type': float, 'range': (0, 1)},
        ...     'batch_size': {'type': int, 'min': 1}
        ... }
        >>> config = {'learning_rate': 0.001, 'batch_size': 32}
        >>> validate_config(config, schema)
        True
    """
    for key, rules in schema.items():
        # Check if required field is present
        if rules.get('required', False) and key not in config:
            raise ValidationException(
                f"Missing required configuration: {key}",
                context={'schema': schema}
            )

        if key in config:
            value = config[key]

            # Check type
            if 'type' in rules and not isinstance(value, rules['type']):
                raise ValidationException(
                    f"Invalid type for {key}: expected {rules['type']}, got {type(value)}",
                    context={'key': key, 'value': value}
                )

            # Check range
            if 'range' in rules:
                min_val, max_val = rules['range']
                if not min_val <= value <= max_val:
                    raise ValidationException(
                        f"Value for {key} out of range: {value} not in [{min_val}, {max_val}]",
                        context={'key': key, 'value': value, 'range': rules['range']}
                    )

            # Check minimum
            if 'min' in rules and value < rules['min']:
                raise ValidationException(
                    f"Value for {key} below minimum: {value} < {rules['min']}",
                    context={'key': key, 'value': value, 'min': rules['min']}
                )

            # Check maximum
            if 'max' in rules and value > rules['max']:
                raise ValidationException(
                    f"Value for {key} above maximum: {value} > {rules['max']}",
                    context={'key': key, 'value': value, 'max': rules['max']}
                )

            # Check allowed values
            if 'choices' in rules and value not in rules['choices']:
                raise ValidationException(
                    f"Invalid value for {key}: {value} not in {rules['choices']}",
                    context={'key': key, 'value': value, 'choices': rules['choices']}
                )

    logger.info("Configuration validation passed")
    return True


def validate_state(
    state: Union[np.ndarray, pd.DataFrame],
    expected_shape: Optional[tuple] = None,
    check_finite: bool = True
) -> bool:
    """
    Validate environment state

    Args:
        state: State array or DataFrame
        expected_shape: Expected shape (optional)
        check_finite: Whether to check for finite values

    Returns:
        True if validation passes

    Raises:
        ValidationException: If validation fails

    Example:
        >>> state = np.random.rand(20, 10)
        >>> validate_state(state, expected_shape=(20, 10))
        True
    """
    # Convert to numpy array if DataFrame
    if isinstance(state, pd.DataFrame):
        state_array = state.values
    else:
        state_array = state

    # Check shape
    if expected_shape and state_array.shape != expected_shape:
        raise ValidationException(
            f"State shape mismatch: expected {expected_shape}, got {state_array.shape}",
            context={'expected': expected_shape, 'actual': state_array.shape}
        )

    # Check for finite values
    if check_finite and not np.isfinite(state_array).all():
        raise ValidationException(
            "State contains non-finite values (NaN or Inf)",
            context={'validation': 'finite_check'}
        )

    logger.debug(f"State validation passed for shape {state_array.shape}")
    return True


def validate_action(
    action: Union[int, float, np.ndarray],
    action_space_type: str,
    action_space_bounds: Optional[tuple] = None,
    n_actions: Optional[int] = None
) -> bool:
    """
    Validate action against action space

    Args:
        action: Action to validate
        action_space_type: Type of action space ('discrete' or 'continuous')
        action_space_bounds: Bounds for continuous actions (min, max)
        n_actions: Number of discrete actions

    Returns:
        True if validation passes

    Raises:
        ValidationException: If validation fails

    Example:
        >>> validate_action(2, 'discrete', n_actions=3)
        True
        >>> validate_action(0.5, 'continuous', action_space_bounds=(-1, 1))
        True
    """
    if action_space_type == 'discrete':
        if n_actions is None:
            raise ValidationException(
                "n_actions must be specified for discrete action spaces",
                context={'action_space_type': action_space_type}
            )

        if not isinstance(action, (int, np.integer)):
            raise ValidationException(
                f"Discrete action must be integer, got {type(action)}",
                context={'action': action}
            )

        if not 0 <= action < n_actions:
            raise ValidationException(
                f"Action {action} out of range [0, {n_actions})",
                context={'action': action, 'n_actions': n_actions}
            )

    elif action_space_type == 'continuous':
        if action_space_bounds is None:
            raise ValidationException(
                "action_space_bounds must be specified for continuous action spaces",
                context={'action_space_type': action_space_type}
            )

        min_val, max_val = action_space_bounds

        # Handle array actions
        if isinstance(action, np.ndarray):
            if not np.all((action >= min_val) & (action <= max_val)):
                raise ValidationException(
                    f"Action values out of bounds [{min_val}, {max_val}]",
                    context={'action': action.tolist(), 'bounds': action_space_bounds}
                )
        else:
            if not min_val <= action <= max_val:
                raise ValidationException(
                    f"Action {action} out of bounds [{min_val}, {max_val}]",
                    context={'action': action, 'bounds': action_space_bounds}
                )

    else:
        raise ValidationException(
            f"Unknown action space type: {action_space_type}",
            context={'valid_types': ['discrete', 'continuous']}
        )

    logger.debug(f"Action validation passed: {action}")
    return True
