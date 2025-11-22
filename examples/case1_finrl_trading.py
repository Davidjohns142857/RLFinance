"""
Case 1: FinRL-style Multi-Asset Trading with DQN

This example demonstrates a complete RL trading system inspired by FinRL:
- Multi-asset portfolio management
- Technical indicators as features
- DQN agent with experience replay
- Comprehensive backtesting

References:
    - FinRL: https://github.com/AI4Finance-Foundation/FinRL
    - Paper: NeurIPS 2020 DRL Workshop

Usage:
    python examples/case1_finrl_trading.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_trading_system.core import TradingConfig
from rl_trading_system.environment.stock_trading_env import StockTradingEnv
from rl_trading_system.agents.dqn_agent import DQNAgent
from rl_trading_system.data.feature_engineer import FeatureEngineer
from rl_trading_system.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(level='INFO', log_dir='./logs', log_to_console=True)
logger = get_logger(__name__)


def generate_synthetic_data(
    symbols: list,
    n_days: int = 500,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic market data for demonstration

    Args:
        symbols: List of stock symbols
        n_days: Number of trading days
        seed: Random seed

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)

    data_list = []

    for symbol in symbols:
        # Generate price data with trend and volatility
        base_price = np.random.uniform(50, 200)
        trend = np.random.uniform(-0.001, 0.002)
        volatility = np.random.uniform(0.01, 0.03)

        prices = [base_price]
        for _ in range(n_days - 1):
            change = trend + np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        prices = np.array(prices)

        # Generate OHLCV
        for i in range(n_days):
            daily_volatility = volatility * prices[i]
            high = prices[i] + np.random.uniform(0, daily_volatility)
            low = prices[i] - np.random.uniform(0, daily_volatility)
            open_price = prices[i-1] if i > 0 else prices[i]
            close = prices[i]
            volume = np.random.uniform(1e6, 1e7)

            data_list.append({
                'timestamp': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
                'symbol': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

    df = pd.DataFrame(data_list)
    logger.info(f"Generated {len(df)} rows of synthetic data for {len(symbols)} symbols")

    return df


def train_dqn_agent(env, agent, episodes: int = 100):
    """
    Train DQN agent on trading environment

    Args:
        env: Trading environment
        agent: DQN agent
        episodes: Number of training episodes

    Returns:
        Training history
    """
    logger.info(f"Starting training for {episodes} episodes")

    history = {
        'episode': [],
        'total_reward': [],
        'portfolio_value': [],
        'epsilon': [],
        'loss': []
    }

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        losses = []

        while not done:
            # Select action
            action = agent.act(state, training=True)

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            if len(agent.memory) > agent.batch_size:
                metrics = agent.train()
                losses.append(metrics['loss'])

            state = next_state
            total_reward += reward
            step += 1

        # Episode statistics
        stats = env.get_portfolio_stats()
        history['episode'].append(episode)
        history['total_reward'].append(total_reward)
        history['portfolio_value'].append(stats.get('final_value', 0))
        history['epsilon'].append(agent.epsilon)
        history['loss'].append(np.mean(losses) if losses else 0)

        if (episode + 1) % 10 == 0:
            logger.info(
                f"Episode {episode + 1}/{episodes}: "
                f"Reward={total_reward:.2f}, "
                f"Value=${stats.get('final_value', 0):,.2f}, "
                f"Sharpe={stats.get('sharpe_ratio', 0):.2f}, "
                f"Epsilon={agent.epsilon:.3f}"
            )

    return history


def backtest_agent(env, agent):
    """
    Backtest trained agent

    Args:
        env: Trading environment
        agent: Trained DQN agent

    Returns:
        Backtest results
    """
    logger.info("Running backtest...")

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Use trained policy (no exploration)
        action = agent.act(state, training=False)

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward

    # Get final statistics
    stats = env.get_portfolio_stats()

    logger.info("=" * 60)
    logger.info("Backtest Results:")
    logger.info(f"  Total Return: {stats['total_return']*100:.2f}%")
    logger.info(f"  Final Value: ${stats['final_value']:,.2f}")
    logger.info(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {stats['max_drawdown']*100:.2f}%")
    logger.info(f"  Volatility: {stats['volatility']*100:.2f}%")
    logger.info(f"  Total Trades: {stats['total_trades']}")
    logger.info("=" * 60)

    return stats


def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("Case 1: FinRL-style Multi-Asset Trading with DQN")
    logger.info("=" * 80)

    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    initial_capital = 100000
    n_days = 500
    train_episodes = 100

    # Generate data
    logger.info("\nStep 1: Generating synthetic market data...")
    data = generate_synthetic_data(symbols, n_days=n_days)

    # Feature engineering
    logger.info("\nStep 2: Calculating technical indicators...")
    feature_engineer = FeatureEngineer({
        'indicators': ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS'],
        'params': {
            'SMA': {'period': 20},
            'EMA': {'period': 12},
            'RSI': {'period': 14},
            'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
            'BBANDS': {'period': 20, 'std': 2}
        }
    })

    # Process each symbol
    data_with_features = []
    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol].copy()
        symbol_data = feature_engineer.process(symbol_data)
        data_with_features.append(symbol_data)

    data = pd.concat(data_with_features, ignore_index=True)
    logger.info(f"Features added: {data.shape[1]} columns")

    # Split data
    train_size = int(len(data) * 0.7 / len(symbols))
    train_data = data[data['timestamp'] <= data['timestamp'].unique()[train_size * len(symbols)]]
    test_data = data[data['timestamp'] > data['timestamp'].unique()[train_size * len(symbols)]]

    logger.info(f"Train data: {len(train_data)} rows")
    logger.info(f"Test data: {len(test_data)} rows")

    # Create environment
    logger.info("\nStep 3: Creating trading environment...")
    tech_indicators = feature_engineer.get_feature_names()

    env_config = {
        'data': train_data,
        'symbols': symbols,
        'initial_capital': initial_capital,
        'transaction_cost': 0.001,
        'slippage': 0.0005,
        'action_type': 'discrete',
        'tech_indicators': tech_indicators
    }

    train_env = StockTradingEnv(env_config)
    logger.info(f"Environment created: {train_env.observation_space.shape}")

    # Create agent
    logger.info("\nStep 4: Creating DQN agent...")
    agent_config = {
        'state_dim': train_env.observation_space.shape[0],
        'action_dim': 3,  # 0: sell, 1: hold, 2: buy
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'batch_size': 32,
        'memory_size': 10000,
        'target_update_freq': 100,
        'hidden_dims': [128, 128],
        'double_dqn': True,
        'dueling_dqn': True
    }

    agent = DQNAgent(agent_config)
    logger.info(f"Agent created: {agent.policy_net}")

    # Train agent
    logger.info("\nStep 5: Training agent...")
    history = train_dqn_agent(train_env, agent, episodes=train_episodes)

    # Create test environment
    logger.info("\nStep 6: Creating test environment...")
    test_env_config = env_config.copy()
    test_env_config['data'] = test_data
    test_env = StockTradingEnv(test_env_config)

    # Backtest
    logger.info("\nStep 7: Backtesting on test data...")
    backtest_results = backtest_agent(test_env, agent)

    # Save model
    model_path = './models/case1_dqn_model.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)
    logger.info(f"\nModel saved to {model_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Case 1 completed successfully!")
    logger.info("=" * 80)

    return history, backtest_results


if __name__ == '__main__':
    main()
