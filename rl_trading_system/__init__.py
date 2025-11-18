"""
RL Trading System

A modular reinforcement learning trading system framework.
"""

__version__ = "1.0.0"
__author__ = "Claude Code"

from rl_trading_system.core import (
    TradingConfig,
    BaseModule,
    BaseStrategy,
    BaseAgent
)

__all__ = [
    'TradingConfig',
    'BaseModule',
    'BaseStrategy',
    'BaseAgent',
]
