"""
RL Trading System - Utils Module

Utility functions and classes for the trading system.
"""

# Import logger (no external dependencies)
from rl_trading_system.utils.logger import (
    setup_logging,
    get_logger,
    LogContext,
    PerformanceLogger,
    AuditLogger
)

__version__ = "1.0.0"

# Validators are imported lazily to avoid requiring pandas/numpy at import time
_validators_module = None


def _get_validators():
    """Lazy import of validators module"""
    global _validators_module
    if _validators_module is None:
        from rl_trading_system.utils import validators as _validators_module
    return _validators_module


def __getattr__(name):
    """Support lazy loading of validator functions"""
    validator_names = [
        'validate_ohlcv',
        'validate_features',
        'validate_order',
        'validate_position',
        'validate_config',
        'validate_state',
        'validate_action'
    ]

    if name in validator_names:
        validators = _get_validators()
        return getattr(validators, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Logging (always available)
    'setup_logging',
    'get_logger',
    'LogContext',
    'PerformanceLogger',
    'AuditLogger',

    # Validators (lazy loaded)
    'validate_ohlcv',
    'validate_features',
    'validate_order',
    'validate_position',
    'validate_config',
    'validate_state',
    'validate_action',
]
