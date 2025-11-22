"""
Module: environment.sharpe_reward_env
Description: Trading environment optimizing Sharpe Ratio (TDQN style)
Author: Claude Code
Date: 2025-11-18
Version: 1.0.0

References:
    - Paper: "An Application of Deep Reinforcement Learning to Algorithmic Trading"
      (arxiv:2004.06627)
    - TDQN: Trading Deep Q-Network optimizing Sharpe ratio

This environment extends StockTradingEnv with Sharpe ratio optimization.

Key Features:
    - Sharpe ratio as primary reward signal
    - Risk-adjusted performance optimization
    - Rolling window for Sharpe calculation
    - Support for both continuous and discrete actions

Example:
    >>> env = SharpeRewardEnv({
    ...     'data': market_data,
    ...     'symbols': ['AAPL'],
    ...     'initial_capital': 100000,
    ...     'sharpe_window': 20
    ... })
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging

from rl_trading_system.environment.stock_trading_env import StockTradingEnv
from rl_trading_system.utils.logger import get_logger

logger = get_logger(__name__)


class SharpeRewardEnv(StockTradingEnv):
    """
    Trading environment with Sharpe ratio reward

    Based on TDQN paper (arxiv:2004.06627), this environment uses
    Sharpe ratio as the primary reward signal to encourage risk-adjusted returns.

    Attributes:
        sharpe_window: Rolling window size for Sharpe calculation
        returns_history: Historical returns for Sharpe computation
        risk_free_rate: Annualized risk-free rate

    Example:
        >>> config = {
        ...     'data': df,
        ...     'symbols': ['BTC/USDT'],
        ...     'initial_capital': 10000,
        ...     'sharpe_window': 20,
        ...     'risk_free_rate': 0.02
        ... }
        >>> env = SharpeRewardEnv(config)
        >>> obs, info = env.reset()
        >>> action = [0.5]  # 50% allocation
        >>> obs, reward, done, truncated, info = env.step(action)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Sharpe reward environment

        Args:
            config: Configuration dictionary (extends StockTradingEnv)
                - sharpe_window (int): Window size for Sharpe calculation
                - risk_free_rate (float): Annual risk-free rate (default: 0.02)
                - sharpe_scaling (float): Scaling factor for Sharpe reward
        """
        # Additional config for Sharpe calculation
        self.sharpe_window = config.get('sharpe_window', 20)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.sharpe_scaling = config.get('sharpe_scaling', 10.0)

        # Initialize base environment
        super().__init__(config)

        # Returns history for Sharpe calculation
        self.returns_history = []

        logger.info(
            f"SharpeRewardEnv initialized: window={self.sharpe_window}, "
            f"risk_free_rate={self.risk_free_rate}"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment and clear returns history"""
        self.returns_history = []
        return super().reset(seed=seed, options=options)

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on Sharpe ratio

        Formula:
            Sharpe Ratio = E[R - Rf] / σ(R - Rf)
            where R = returns, Rf = risk-free rate

        Returns:
            Sharpe ratio-based reward
        """
        # Calculate period return
        period_return = (
            (self.portfolio_value - self.previous_portfolio_value)
            / self.previous_portfolio_value
        )

        # Add to history
        self.returns_history.append(period_return)

        # Calculate Sharpe ratio if we have enough history
        if len(self.returns_history) >= self.sharpe_window:
            # Use recent returns
            recent_returns = np.array(self.returns_history[-self.sharpe_window:])

            # Adjust for risk-free rate (assuming daily returns, annualized rf)
            daily_rf = self.risk_free_rate / 252
            excess_returns = recent_returns - daily_rf

            # Calculate Sharpe ratio
            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns)

            if std_excess > 1e-8:
                sharpe_ratio = mean_excess / std_excess
                # Annualize
                sharpe_ratio *= np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            # Use Sharpe ratio as reward (scaled)
            reward = sharpe_ratio * self.sharpe_scaling
        else:
            # Not enough history: use simple return
            reward = period_return * 100  # Scale up simple returns

        return reward

    def get_portfolio_stats(self) -> Dict:
        """
        Get portfolio statistics including Sharpe ratio

        Returns:
            Performance metrics dictionary
        """
        stats = super().get_portfolio_stats()

        # Calculate realized Sharpe ratio from all returns
        if len(self.returns_history) > 1:
            returns_array = np.array(self.returns_history)
            daily_rf = self.risk_free_rate / 252
            excess_returns = returns_array - daily_rf

            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns)

            if std_excess > 1e-8:
                realized_sharpe = (mean_excess / std_excess) * np.sqrt(252)
            else:
                realized_sharpe = 0.0

            stats['realized_sharpe'] = realized_sharpe
            stats['avg_return'] = np.mean(returns_array) * 252  # Annualized
            stats['volatility'] = np.std(returns_array) * np.sqrt(252)  # Annualized
        else:
            stats['realized_sharpe'] = 0.0
            stats['avg_return'] = 0.0
            stats['volatility'] = 0.0

        return stats
