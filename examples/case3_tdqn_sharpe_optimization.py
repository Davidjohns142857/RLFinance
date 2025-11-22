"""
Case 3: TDQN - Trading DQN with Sharpe Ratio Optimization

This example demonstrates Sharpe ratio optimization:
- Sharpe ratio as reward signal
- Risk-adjusted performance focus
- Rolling window Sharpe calculation

References:
    - Paper: "An Application of Deep Reinforcement Learning to Algorithmic Trading"
      (arxiv:2004.06627)

Usage:
    python examples/case3_tdqn_sharpe_optimization.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_trading_system.environment.sharpe_reward_env import SharpeRewardEnv
from rl_trading_system.agents.dqn_agent import DQNAgent
from rl_trading_system.data.feature_engineer import FeatureEngineer
from rl_trading_system.utils.logger import setup_logging, get_logger

setup_logging(level='INFO', log_dir='./logs', log_to_console=True)
logger = get_logger(__name__)


def generate_volatile_data(n_days: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate volatile market data to test Sharpe optimization"""
    np.random.seed(seed)

    # Create regime-switching market
    regimes = []
    current_regime = 'bull'
    days_in_regime = 0

    for i in range(n_days):
        if days_in_regime > np.random.randint(50, 150):
            # Switch regime
            current_regime = 'bear' if current_regime == 'bull' else 'bull'
            days_in_regime = 0
        regimes.append(current_regime)
        days_in_regime += 1

    # Generate prices based on regime
    base_price = 100
    prices = [base_price]

    for regime in regimes[1:]:
        if regime == 'bull':
            change = np.random.normal(0.001, 0.02)  # Positive trend, moderate vol
        else:
            change = np.random.normal(-0.001, 0.04)  # Negative trend, high vol

        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 10))

    prices = np.array(prices)

    data = []
    for i in range(n_days):
        daily_vol = 0.02 * prices[i]
        data.append({
            'timestamp': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
            'symbol': 'VOL/USDT',
            'open': prices[i-1] if i > 0 else prices[i],
            'high': prices[i] + np.random.uniform(0, daily_vol),
            'low': prices[i] - np.random.uniform(0, daily_vol),
            'close': prices[i],
            'volume': np.random.uniform(1e6, 1e7),
            'regime': regimes[i]
        })

    return pd.DataFrame(data)


def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("Case 3: TDQN - Sharpe Ratio Optimization")
    logger.info("=" * 80)

    # Generate regime-switching data
    logger.info("\nGenerating volatile market data...")
    data = generate_volatile_data(n_days=1000)

    # Feature engineering
    logger.info("Calculating technical indicators...")
    engineer = FeatureEngineer({
        'indicators': ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'ATR'],
        'params': {
            'SMA': {'period': 20},
            'EMA': {'period': 12},
            'RSI': {'period': 14},
            'ATR': {'period': 14}
        }
    })

    data = engineer.process(data)
    indicators = engineer.get_feature_names()

    # Split data
    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()

    # Create Sharpe-optimizing environment
    logger.info("\nCreating Sharpe reward environment...")
    env_config = {
        'data': train_data,
        'symbols': ['VOL/USDT'],
        'initial_capital': 10000,
        'transaction_cost': 0.001,
        'action_type': 'continuous',  # Continuous actions for better control
        'tech_indicators': indicators,
        'sharpe_window': 20,
        'risk_free_rate': 0.02,
        'sharpe_scaling': 10.0
    }

    train_env = SharpeRewardEnv(env_config)

    # Create agent
    logger.info("Creating DQN agent for continuous actions...")
    # For continuous actions, we'll discretize the action space
    # -1 to 1 -> 5 discrete levels: [-1, -0.5, 0, 0.5, 1]

    agent = DQNAgent({
        'state_dim': train_env.observation_space.shape[0],
        'action_dim': 5,  # 5 position levels
        'learning_rate': 0.0003,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.998,
        'epsilon_min': 0.01,
        'batch_size': 64,
        'memory_size': 20000,
        'double_dqn': True,
        'dueling_dqn': True
    })

    # Action mapping: 0->-1, 1->-0.5, 2->0, 3->0.5, 4->1
    action_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}

    # Train
    logger.info("\nTraining TDQN for 100 episodes...")
    for episode in range(100):
        state, _ = train_env.reset()
        done = False
        total_reward = 0

        while not done:
            # Get discrete action
            discrete_action = agent.act(state, training=True)

            # Map to continuous action
            continuous_action = np.array([action_map[discrete_action]])

            # Execute
            next_state, reward, terminated, truncated, _ = train_env.step(continuous_action)
            done = terminated or truncated

            # Store transition with discrete action
            agent.remember(state, discrete_action, reward, next_state, done)

            if len(agent.memory) > agent.batch_size:
                agent.train()

            state = next_state
            total_reward += reward

        if (episode + 1) % 10 == 0:
            stats = train_env.get_portfolio_stats()
            logger.info(
                f"Episode {episode + 1}: "
                f"Reward={total_reward:.2f}, "
                f"Sharpe={stats.get('realized_sharpe', 0):.2f}, "
                f"Return={stats['total_return']*100:.2f}%"
            )

    # Test
    logger.info("\nTesting on hold-out data...")
    test_env_config = env_config.copy()
    test_env_config['data'] = test_data
    test_env = SharpeRewardEnv(test_env_config)

    state, _ = test_env.reset()
    done = False

    while not done:
        discrete_action = agent.act(state, training=False)
        continuous_action = np.array([action_map[discrete_action]])
        state, _, terminated, truncated, _ = test_env.step(continuous_action)
        done = terminated or truncated

    stats = test_env.get_portfolio_stats()

    logger.info("\n" + "=" * 60)
    logger.info("TDQN Test Results (Sharpe-Optimized):")
    logger.info(f"  Total Return: {stats['total_return']*100:.2f}%")
    logger.info(f"  Final Value: ${stats['final_value']:,.2f}")
    logger.info(f"  Realized Sharpe: {stats.get('realized_sharpe', 0):.2f}")
    logger.info(f"  Annualized Return: {stats.get('avg_return', 0)*100:.2f}%")
    logger.info(f"  Annualized Vol: {stats.get('volatility', 0)*100:.2f}%")
    logger.info(f"  Max Drawdown: {stats['max_drawdown']*100:.2f}%")
    logger.info(f"  Total Trades: {stats['total_trades']}")
    logger.info("=" * 60)

    # Compare with standard Sharpe
    logger.info("\nSharpe Ratio Comparison:")
    logger.info(f"  Realized Sharpe: {stats.get('realized_sharpe', 0):.2f}")
    logger.info(f"  Standard Sharpe: {stats.get('sharpe_ratio', 0):.2f}")

    # Save model
    agent.save('./models/case3_tdqn_model.pth')
    logger.info("\nModel saved!")

    logger.info("\nCase 3 completed!")


if __name__ == '__main__':
    main()
