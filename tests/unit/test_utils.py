"""
Unit tests for utils module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rl_trading_system.utils import (
    validate_ohlcv,
    validate_features,
    validate_order,
    validate_position,
    validate_config,
    validate_state,
    validate_action
)
from rl_trading_system.core.exceptions import DataValidationException, ValidationException


class TestOHLCVValidation:
    """Test OHLCV data validation"""

    def test_valid_ohlcv(self):
        """Test validation of valid OHLCV data"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10),
            'open': np.random.rand(10) * 100 + 100,
            'high': np.random.rand(10) * 100 + 150,
            'low': np.random.rand(10) * 100 + 50,
            'close': np.random.rand(10) * 100 + 100,
            'volume': np.random.rand(10) * 1000
        })
        # Ensure price relationships are valid
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        assert validate_ohlcv(df) is True

    def test_empty_dataframe(self):
        """Test validation of empty DataFrame"""
        df = pd.DataFrame()
        with pytest.raises(DataValidationException):
            validate_ohlcv(df)

    def test_missing_columns(self):
        """Test validation with missing columns"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10),
            'open': np.random.rand(10) * 100
        })
        with pytest.raises(DataValidationException):
            validate_ohlcv(df)


class TestFeatureValidation:
    """Test feature validation"""

    def test_valid_features(self):
        """Test validation of valid features"""
        df = pd.DataFrame(np.random.rand(100, 10))
        assert validate_features(df) is True

    def test_features_with_inf(self):
        """Test validation with infinite values"""
        df = pd.DataFrame(np.random.rand(100, 10))
        df.iloc[0, 0] = np.inf
        with pytest.raises(DataValidationException):
            validate_features(df, allow_inf=False)

    def test_features_with_nan(self):
        """Test validation with NaN values"""
        df = pd.DataFrame(np.random.rand(100, 10))
        df.iloc[:50, 0] = np.nan  # 50% NaN in first column
        with pytest.raises(DataValidationException):
            validate_features(df, allow_nan=False, max_nan_ratio=0.1)


class TestOrderValidation:
    """Test order validation"""

    def test_valid_market_order(self):
        """Test validation of valid market order"""
        order = {
            'symbol': 'BTC/USDT',
            'type': 'market',
            'side': 'buy',
            'quantity': 1.0
        }
        assert validate_order(order) is True

    def test_valid_limit_order(self):
        """Test validation of valid limit order"""
        order = {
            'symbol': 'BTC/USDT',
            'type': 'limit',
            'side': 'sell',
            'quantity': 1.5,
            'price': 50000.0
        }
        assert validate_order(order) is True

    def test_missing_fields(self):
        """Test validation with missing fields"""
        order = {
            'symbol': 'BTC/USDT',
            'type': 'market'
        }
        with pytest.raises(ValidationException):
            validate_order(order)

    def test_invalid_order_type(self):
        """Test validation with invalid order type"""
        order = {
            'symbol': 'BTC/USDT',
            'type': 'invalid',
            'side': 'buy',
            'quantity': 1.0
        }
        with pytest.raises(ValidationException):
            validate_order(order)


class TestPositionValidation:
    """Test position validation"""

    def test_valid_position(self):
        """Test validation of valid position"""
        position = {
            'symbol': 'BTC/USDT',
            'quantity': 1.5,
            'entry_price': 50000.0,
            'current_price': 51000.0
        }
        assert validate_position(position) is True

    def test_zero_quantity(self):
        """Test validation with zero quantity"""
        position = {
            'symbol': 'BTC/USDT',
            'quantity': 0,
            'entry_price': 50000.0
        }
        with pytest.raises(ValidationException):
            validate_position(position)


class TestConfigValidation:
    """Test config validation"""

    def test_valid_config(self):
        """Test validation of valid config"""
        schema = {
            'learning_rate': {'type': float, 'range': (0, 1)},
            'batch_size': {'type': int, 'min': 1}
        }
        config = {'learning_rate': 0.001, 'batch_size': 32}
        assert validate_config(config, schema) is True

    def test_invalid_type(self):
        """Test validation with invalid type"""
        schema = {
            'learning_rate': {'type': float}
        }
        config = {'learning_rate': "0.001"}  # String instead of float
        with pytest.raises(ValidationException):
            validate_config(config, schema)


class TestStateValidation:
    """Test state validation"""

    def test_valid_state(self):
        """Test validation of valid state"""
        state = np.random.rand(20, 10)
        assert validate_state(state, expected_shape=(20, 10)) is True

    def test_invalid_shape(self):
        """Test validation with invalid shape"""
        state = np.random.rand(20, 10)
        with pytest.raises(ValidationException):
            validate_state(state, expected_shape=(20, 5))


class TestActionValidation:
    """Test action validation"""

    def test_valid_discrete_action(self):
        """Test validation of valid discrete action"""
        assert validate_action(2, 'discrete', n_actions=3) is True

    def test_invalid_discrete_action(self):
        """Test validation of invalid discrete action"""
        with pytest.raises(ValidationException):
            validate_action(5, 'discrete', n_actions=3)

    def test_valid_continuous_action(self):
        """Test validation of valid continuous action"""
        assert validate_action(0.5, 'continuous', action_space_bounds=(-1, 1)) is True

    def test_invalid_continuous_action(self):
        """Test validation of invalid continuous action"""
        with pytest.raises(ValidationException):
            validate_action(2.0, 'continuous', action_space_bounds=(-1, 1))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
