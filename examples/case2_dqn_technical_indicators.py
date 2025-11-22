"""
Case 2: DQN Trading with Technical Indicators

This example demonstrates DQN trading using technical indicators as state:
- Simple state representation (MA, RSI)
- Discrete actions (Buy, Sell, Hold)
- Direct profit optimization

References:
    - Paper: "Quantitative Trading using Deep Q Learning" (arxiv:2304.06037)

Usage:
    python examples/case2_dqn_technical_indicators.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_trading_system.environment.stock_trading_env import StockTradingEnv
from rl_trading_system.agents.dqn_agent import DQNAgent
from rl_trading_system.data.feature_engineer import FeatureEngineer
from rl_trading_system.utils.logger import setup_logging, get_logger

setup_logging(level='INFO', log_dir='./logs', log_to_console=True)
logger = get_logger(__name__)


def generate_btc_data(n_days: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Bitcoin-like price data"""
    np.random.seed(seed)

    # Bitcoin-like characteristics: high volatility, trending
    base_price = 10000
    trend = 0.002  # Upward trend
    volatility = 0.05  # 5% daily volatility

    prices = [base_price]
    for _ in range(n_days - 1):
        change = trend + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 100))  # Floor price

    prices = np.array(prices)

    data = []
    for i in range(n_days):
        daily_vol = volatility * prices[i]
        data.append({
            'timestamp': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
            'symbol': 'BTC/USDT',
            'open': prices[i-1] if i > 0 else prices[i],
            'high': prices[i] + np.random.uniform(0, daily_vol),
            'low': prices[i] - np.random.uniform(0, daily_vol),
            'close': prices[i],
            'volume': np.random.uniform(1e6, 1e8)
        })

    return pd.DataFrame(data)


def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("Case 2: DQN Trading with Technical Indicators")
    logger.info("=" * 80)

    # Generate data
    logger.info("\nGenerating BTC/USDT data...")
    data = generate_btc_data(n_days=1000)

    # Calculate technical indicators
    logger.info("Calculating technical indicators...")
    engineer = FeatureEngineer({
        'indicators': ['SMA', 'EMA', 'RSI', 'MACD'],
        'params': {
            'SMA': {'period': 20},
            'EMA': {'period': 12},
            'RSI': {'period': 14}
        }
    })

    data = engineer.process(data)
    indicators = engineer.get_feature_names()

    # Split data
    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()

    logger.info(f"Train: {len(train_data)} days, Test: {len(test_data)} days")

    # Create environment (single asset)
    logger.info("\nCreating trading environment...")
    env_config = {
        'data': train_data,
        'symbols': ['BTC/USDT'],
        'initial_capital': 10000,
        'transaction_cost': 0.001,
        'action_type': 'discrete',
        'tech_indicators': indicators
    }

    train_env = StockTradingEnv(env_config)

    # Create DQN agent
    logger.info("Creating DQN agent...")
    agent = DQNAgent({
        'state_dim': train_env.observation_space.shape[0],
        'action_dim': 3,
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'batch_size': 64,
        'memory_size': 10000,
        'double_dqn': True
    })

    # Train
    logger.info("\nTraining for 50 episodes...")
    for episode in range(50):
        state, _ = train_env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > agent.batch_size:
                agent.train()

            state = next_state
            total_reward += reward

        if (episode + 1) % 10 == 0:
            stats = train_env.get_portfolio_stats()
            logger.info(
                f"Episode {episode + 1}: "
                f"Return={stats['total_return']*100:.2f}%, "
                f"Sharpe={stats.get('sharpe_ratio', 0):.2f}"
            )

    # Test
    logger.info("\nTesting on hold-out data...")
    test_env_config = env_config.copy()
    test_env_config['data'] = test_data
    test_env = StockTradingEnv(test_env_config)

    state, _ = test_env.reset()
    done = False

    while not done:
        action = agent.act(state, training=False)
        state, _, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated

    stats = test_env.get_portfolio_stats()

    logger.info("\n" + "=" * 60)
    logger.info("Test Results:")
    logger.info(f"  Total Return: {stats['total_return']*100:.2f}%")
    logger.info(f"  Final Value: ${stats['final_value']:,.2f}")
    logger.info(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {stats['max_drawdown']*100:.2f}%")
    logger.info(f"  Total Trades: {stats['total_trades']}")
    logger.info("=" * 60)

    # Save model
    agent.save('./models/case2_dqn_model.pth')
    logger.info("\nModel saved!")

    logger.info("\nCase 2 completed!")


if __name__ == '__main__':
    main()
