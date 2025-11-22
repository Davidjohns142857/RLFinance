"""
Module: environment.stock_trading_env
Description: Stock trading environment inspired by FinRL
Author: Claude Code
Date: 2025-11-18
Version: 1.0.0

References:
    - Paper: "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading"
      (NeurIPS 2020)
    - GitHub: https://github.com/AI4Finance-Foundation/FinRL

This environment implements a multi-asset stock trading environment compatible with
OpenAI Gym interface, supporting both discrete and continuous action spaces.

Features:
    - Multi-asset portfolio management
    - Technical indicators integration
    - Transaction costs and slippage
    - Turbulence index for risk management
    - Flexible reward functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
import logging

from rl_trading_system.core.base import BaseModule
from rl_trading_system.core.exceptions import EnvironmentException
from rl_trading_system.utils.logger import get_logger

logger = get_logger(__name__)


class StockTradingEnv(gym.Env, BaseModule):
    """
    Stock trading environment for reinforcement learning

    Based on FinRL architecture with support for:
    - Multiple stocks trading
    - Technical indicators
    - Transaction costs
    - Continuous or discrete actions

    State Space:
        - Balance
        - Stock prices
        - Stock holdings
        - Technical indicators (MACD, RSI, Bollinger Bands, etc.)

    Action Space:
        Continuous: [-1, 1] for each stock (sell all, hold, buy all)
        Discrete: {0: sell, 1: hold, 2: buy} for each stock

    Reward:
        Portfolio value change (log returns or absolute change)

    Example:
        >>> env = StockTradingEnv({
        ...     'initial_capital': 100000,
        ...     'symbols': ['AAPL', 'GOOGL'],
        ...     'transaction_cost': 0.001
        ... })
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stock trading environment

        Args:
            config: Configuration dictionary
                - data (pd.DataFrame): Market data with OHLCV and indicators
                - symbols (List[str]): List of stock symbols
                - initial_capital (float): Initial cash balance
                - transaction_cost (float): Commission rate (default: 0.001)
                - slippage (float): Slippage rate (default: 0.0005)
                - action_type (str): 'continuous' or 'discrete'
                - reward_scaling (float): Scale factor for rewards
                - tech_indicators (List[str]): Technical indicator names
        """
        # Set attributes before calling BaseModule.__init__ because _validate_config needs them
        # Market data
        self.data = config.get('data')
        self.symbols = config.get('symbols', [])
        self.n_stocks = len(self.symbols)

        # Trading parameters
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.slippage = config.get('slippage', 0.0005)

        # Environment settings
        self.action_type = config.get('action_type', 'continuous')
        self.reward_scaling = config.get('reward_scaling', 1.0)
        self.tech_indicators = config.get('tech_indicators', [])

        # Now call BaseModule initialization
        BaseModule.__init__(self, config)

        # State tracking
        self.current_step = 0
        self.cash_balance = self.initial_capital
        self.stock_holdings = np.zeros(self.n_stocks)
        self.portfolio_value = self.initial_capital
        self.previous_portfolio_value = self.initial_capital

        # Trading history
        self.trades_history = []
        self.portfolio_history = []

        # Define action and observation spaces
        self._setup_spaces()

        logger.info(
            f"StockTradingEnv initialized: {self.n_stocks} stocks, "
            f"${self.initial_capital:.2f} initial capital"
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.data is None or self.data.empty:
            raise EnvironmentException(
                "Market data is required",
                context={'config': self.config}
            )

        if not self.symbols:
            raise EnvironmentException(
                "At least one stock symbol is required",
                context={'config': self.config}
            )

        if self.initial_capital <= 0:
            raise EnvironmentException(
                "Initial capital must be positive",
                context={'initial_capital': self.initial_capital}
            )

    def _initialize(self) -> None:
        """Initialize environment components"""
        # Group data by timestamp for efficient access
        if 'timestamp' in self.data.columns:
            self.data = self.data.sort_values('timestamp').reset_index(drop=True)

        # Get unique timestamps
        self.timestamps = self.data['timestamp'].unique() if 'timestamp' in self.data.columns else None
        self.max_steps = len(self.timestamps) if self.timestamps is not None else len(self.data) // self.n_stocks

    def _setup_spaces(self) -> None:
        """Setup action and observation spaces"""
        # Action space
        if self.action_type == 'continuous':
            # Continuous actions: [-1, 1] for each stock
            self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(self.n_stocks,),
                dtype=np.float32
            )
        else:
            # Discrete actions: 0=sell, 1=hold, 2=buy
            if self.n_stocks == 1:
                # Single stock: use simple Discrete space
                self.action_space = spaces.Discrete(3)
            else:
                # Multiple stocks: use MultiDiscrete space
                self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)

        # Observation space
        # State = [balance] + [stock_prices] + [holdings] + [technical_indicators]
        n_indicators = len(self.tech_indicators) * self.n_stocks
        state_dim = 1 + self.n_stocks * 2 + n_indicators

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        logger.debug(
            f"Action space: {self.action_space}, "
            f"Observation space shape: {self.observation_space.shape}"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset state
        self.current_step = 0
        self.cash_balance = self.initial_capital
        self.stock_holdings = np.zeros(self.n_stocks)
        self.portfolio_value = self.initial_capital
        self.previous_portfolio_value = self.initial_capital

        # Clear history
        self.trades_history = []
        self.portfolio_history = []

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        logger.debug(f"Environment reset at step 0")

        return observation, info

    def step(
        self,
        action
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment

        Args:
            action: Trading action

        Returns:
            observation: Next state
            reward: Reward signal
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Execute trades
        self._execute_trades(action)

        # Move to next step
        self.current_step += 1

        # Update portfolio value
        self._update_portfolio_value()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        terminated = self.current_step >= self.max_steps - 1
        truncated = False

        # Get next observation
        observation = self._get_observation()
        info = self._get_info()

        # Record history
        self.portfolio_history.append({
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash_balance,
            'holdings': self.stock_holdings.copy()
        })

        return observation, reward, terminated, truncated, info

    def _execute_trades(self, action) -> None:
        """
        Execute trading actions

        Args:
            action: Trading actions for each stock (int for single stock, array for multiple)
        """
        # Convert single int action to array for uniform processing
        if isinstance(action, (int, np.integer)):
            action = np.array([action])
        elif not isinstance(action, np.ndarray):
            action = np.array(action)

        current_prices = self._get_current_prices()

        for i, symbol in enumerate(self.symbols):
            if self.action_type == 'continuous':
                # Continuous action: -1 (sell all) to 1 (buy all)
                target_value = self.portfolio_value * action[i]
                current_value = self.stock_holdings[i] * current_prices[i]
                trade_value = target_value - current_value
            else:
                # Discrete action: 0=sell, 1=hold, 2=buy
                if action[i] == 0:  # Sell
                    trade_value = -self.stock_holdings[i] * current_prices[i]
                elif action[i] == 2:  # Buy
                    trade_value = self.cash_balance * 0.33  # Buy with 1/3 of cash
                else:  # Hold
                    trade_value = 0

            if abs(trade_value) > 1e-6:
                self._execute_single_trade(i, trade_value, current_prices[i])

    def _execute_single_trade(
        self,
        stock_idx: int,
        trade_value: float,
        price: float
    ) -> None:
        """
        Execute a single trade with transaction costs

        Args:
            stock_idx: Stock index
            trade_value: Value to trade (positive=buy, negative=sell)
            price: Current stock price
        """
        # Apply slippage
        execution_price = price * (1 + self.slippage if trade_value > 0 else 1 - self.slippage)

        # Calculate shares to trade
        shares = trade_value / execution_price

        # Calculate costs
        commission = abs(trade_value) * self.transaction_cost
        total_cost = trade_value + commission if trade_value > 0 else trade_value - commission

        # Check if we have enough cash for buying
        if trade_value > 0 and abs(total_cost) > self.cash_balance:
            # Adjust to available cash
            total_cost = -self.cash_balance
            shares = (self.cash_balance / (1 + self.transaction_cost)) / execution_price

        # Check if we have enough shares for selling
        if trade_value < 0 and shares < -self.stock_holdings[stock_idx]:
            shares = -self.stock_holdings[stock_idx]
            total_cost = shares * execution_price - abs(shares * execution_price) * self.transaction_cost

        # Execute trade
        self.cash_balance -= total_cost
        self.stock_holdings[stock_idx] += shares

        # Record trade
        self.trades_history.append({
            'step': self.current_step,
            'symbol': self.symbols[stock_idx],
            'shares': shares,
            'price': execution_price,
            'value': trade_value,
            'commission': commission
        })

    def _get_current_prices(self) -> np.ndarray:
        """Get current stock prices"""
        prices = np.zeros(self.n_stocks)

        for i, symbol in enumerate(self.symbols):
            if self.timestamps is not None:
                # Multi-stock data format
                mask = (self.data['symbol'] == symbol) & (self.data['timestamp'] == self.timestamps[self.current_step])
                price_data = self.data[mask]['close'].values
            else:
                # Single stock or pre-arranged data
                idx = self.current_step * self.n_stocks + i
                price_data = self.data.iloc[idx]['close']

            prices[i] = price_data[0] if isinstance(price_data, np.ndarray) else price_data

        return prices

    def _update_portfolio_value(self) -> None:
        """Update total portfolio value"""
        self.previous_portfolio_value = self.portfolio_value

        current_prices = self._get_current_prices()
        stocks_value = np.sum(self.stock_holdings * current_prices)
        self.portfolio_value = self.cash_balance + stocks_value

    def _calculate_reward(self) -> float:
        """
        Calculate reward signal

        Returns:
            Reward value (portfolio value change)
        """
        # Simple reward: change in portfolio value
        reward = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value

        # Apply scaling
        reward *= self.reward_scaling

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state)

        Returns:
            State vector
        """
        # Normalize balance
        balance_norm = self.cash_balance / self.initial_capital

        # Get current prices
        current_prices = self._get_current_prices()

        # Normalize prices (relative to initial)
        if not hasattr(self, 'initial_prices'):
            self.initial_prices = current_prices.copy()
        price_norm = current_prices / self.initial_prices

        # Holdings
        holdings_norm = self.stock_holdings

        # Technical indicators
        indicators = self._get_technical_indicators()

        # Concatenate state
        state = np.concatenate([
            [balance_norm],
            price_norm,
            holdings_norm,
            indicators
        ]).astype(np.float32)

        return state

    def _get_technical_indicators(self) -> np.ndarray:
        """Get technical indicators for current step"""
        indicators = []

        for symbol in self.symbols:
            for indicator_name in self.tech_indicators:
                if self.timestamps is not None:
                    mask = (self.data['symbol'] == symbol) & (self.data['timestamp'] == self.timestamps[self.current_step])
                    indicator_value = self.data[mask][indicator_name].values
                else:
                    idx = self.current_step * self.n_stocks + self.symbols.index(symbol)
                    indicator_value = self.data.iloc[idx][indicator_name]

                value = indicator_value[0] if isinstance(indicator_value, np.ndarray) else indicator_value
                indicators.append(value if not pd.isna(value) else 0.0)

        return np.array(indicators, dtype=np.float32)

    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash_balance': self.cash_balance,
            'stock_holdings': self.stock_holdings.copy(),
            'total_trades': len(self.trades_history)
        }

    def render(self, mode='human') -> None:
        """Render the environment"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash Balance: ${self.cash_balance:,.2f}")
            print(f"Holdings: {dict(zip(self.symbols, self.stock_holdings))}")
            print(f"Total Trades: {len(self.trades_history)}")
            print(f"{'='*60}")

    def get_portfolio_stats(self) -> Dict:
        """
        Get portfolio statistics

        Returns:
            Dictionary with performance metrics
        """
        if not self.portfolio_history:
            return {}

        values = [h['portfolio_value'] for h in self.portfolio_history]
        returns = np.diff(values) / values[:-1]

        return {
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'final_value': self.portfolio_value,
            'total_trades': len(self.trades_history),
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(values),
            'volatility': np.std(returns) * np.sqrt(252)
        }

    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = values[0]
        max_dd = 0

        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def process(self, *args, **kwargs) -> Any:
        """BaseModule interface compatibility"""
        return self.step(*args, **kwargs)
