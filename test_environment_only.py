#!/usr/bin/env python
"""
Test script for trading environment without DQN agent
Tests environment and feature engineering independently
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from rl_trading_system.environment.stock_trading_env import StockTradingEnv
from rl_trading_system.environment.sharpe_reward_env import SharpeRewardEnv
from rl_trading_system.data.feature_engineer import FeatureEngineer

def generate_test_data(n_days=200, seed=42):
    """Generate synthetic market data"""
    np.random.seed(seed)
    base_price = 100
    prices = [base_price]

    for _ in range(n_days - 1):
        change = np.random.normal(0.001, 0.02)
        prices.append(prices[-1] * (1 + change))

    data = []
    for i in range(n_days):
        data.append({
            'timestamp': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
            'symbol': 'TEST',
            'open': prices[i],
            'high': prices[i] * 1.01,
            'low': prices[i] * 0.99,
            'close': prices[i],
            'volume': np.random.uniform(1e6, 1e7)
        })

    return pd.DataFrame(data)

def test_feature_engineer():
    """Test feature engineering"""
    print("=" * 60)
    print("Testing FeatureEngineer")
    print("=" * 60)

    df = generate_test_data(200)

    # Test with multiple indicators
    engineer = FeatureEngineer({
        'indicators': ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'ATR'],
        'params': {
            'SMA': {'period': 20},
            'EMA': {'period': 12},
            'RSI': {'period': 14},
            'ATR': {'period': 14}
        }
    })

    df_features = engineer.process(df)
    feature_names = engineer.get_feature_names()

    print(f"✓ Original columns: {list(df.columns)}")
    print(f"✓ Features added: {feature_names}")
    print(f"✓ Total columns: {df_features.shape[1]}")
    print(f"✓ No NaN values: {df_features.isna().sum().sum() == 0}")
    print()

def test_stock_trading_env():
    """Test stock trading environment"""
    print("=" * 60)
    print("Testing StockTradingEnv")
    print("=" * 60)

    df = generate_test_data(200)

    engineer = FeatureEngineer({
        'indicators': ['SMA', 'RSI'],
        'params': {'SMA': {'period': 20}, 'RSI': {'period': 14}}
    })
    df_features = engineer.process(df)

    # Test discrete action space
    env = StockTradingEnv({
        'data': df_features,
        'symbols': ['TEST'],
        'initial_capital': 10000,
        'transaction_cost': 0.001,
        'action_type': 'discrete',
        'tech_indicators': engineer.get_feature_names()
    })

    print(f"✓ Environment created")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space shape: {env.observation_space.shape}")

    # Test episode
    obs, info = env.reset()
    print(f"✓ Reset successful, observation shape: {obs.shape}")

    total_reward = 0
    for step in range(50):
        action = np.random.choice([0, 1, 2])  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    stats = env.get_portfolio_stats()
    print(f"✓ Ran {step + 1} steps")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final value: ${stats['final_value']:,.2f}")
    print(f"  Total return: {stats['total_return']*100:.2f}%")
    print(f"  Sharpe ratio: {stats['sharpe_ratio']:.2f}")
    print()

def test_sharpe_reward_env():
    """Test Sharpe reward environment"""
    print("=" * 60)
    print("Testing SharpeRewardEnv")
    print("=" * 60)

    df = generate_test_data(200)

    engineer = FeatureEngineer({
        'indicators': ['SMA', 'RSI', 'MACD'],
        'params': {'SMA': {'period': 20}, 'RSI': {'period': 14}}
    })
    df_features = engineer.process(df)

    # Test with continuous actions
    env = SharpeRewardEnv({
        'data': df_features,
        'symbols': ['TEST'],
        'initial_capital': 10000,
        'transaction_cost': 0.001,
        'action_type': 'continuous',
        'tech_indicators': engineer.get_feature_names(),
        'sharpe_window': 20,
        'risk_free_rate': 0.02
    })

    print(f"✓ Environment created")
    print(f"  Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"✓ Reset successful")

    total_reward = 0
    for step in range(50):
        action = np.random.uniform(-0.5, 0.5, size=1)  # Random continuous action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    stats = env.get_portfolio_stats()
    print(f"✓ Ran {step + 1} steps")
    print(f"  Total reward (Sharpe-based): {total_reward:.4f}")
    print(f"  Final value: ${stats['final_value']:,.2f}")
    print(f"  Realized Sharpe: {stats.get('realized_sharpe', 0):.2f}")
    print(f"  Annualized return: {stats.get('avg_return', 0)*100:.2f}%")
    print()

if __name__ == '__main__':
    try:
        test_feature_engineer()
        test_stock_trading_env()
        test_sharpe_reward_env()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
