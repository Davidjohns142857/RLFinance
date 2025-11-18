"""
Unit tests for core module
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rl_trading_system.core import (
    TradingConfig,
    DataConfig,
    ModelConfig,
    RiskConfig,
    BaseModule,
    BaseStrategy,
    BaseAgent,
    TradingException,
    DataException,
    RiskException
)


class TestTradingConfig:
    """Test TradingConfig class"""

    def test_default_config(self):
        """Test default configuration"""
        config = TradingConfig()
        assert config.data.source == "binance"
        assert config.model.learning_rate == 0.001
        assert config.risk.max_position_size == 0.1

    def test_config_modification(self):
        """Test configuration modification"""
        config = TradingConfig()
        config.data.symbols = ['BTC/USDT', 'ETH/USDT']
        config.model.learning_rate = 0.0001
        assert len(config.data.symbols) == 2
        assert config.model.learning_rate == 0.0001

    def test_config_to_dict(self):
        """Test configuration to dictionary conversion"""
        config = TradingConfig()
        config_dict = config.to_dict()
        assert 'data' in config_dict
        assert 'model' in config_dict
        assert 'risk' in config_dict

    def test_config_validation(self):
        """Test configuration validation"""
        config = TradingConfig()
        assert config.validate() is True


class TestDataConfig:
    """Test DataConfig class"""

    def test_default_data_config(self):
        """Test default data configuration"""
        config = DataConfig()
        assert config.source == "binance"
        assert config.timeframe == "1h"
        assert config.lookback_period == 365

    def test_data_config_validation(self):
        """Test data configuration validation"""
        with pytest.raises(ValueError):
            DataConfig(symbols=[])  # Empty symbols should raise error

        with pytest.raises(ValueError):
            DataConfig(lookback_period=-1)  # Negative lookback should raise error


class TestModelConfig:
    """Test ModelConfig class"""

    def test_default_model_config(self):
        """Test default model configuration"""
        config = ModelConfig()
        assert config.model_type == "dqn"
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.gamma == 0.99

    def test_model_config_validation(self):
        """Test model configuration validation"""
        with pytest.raises(ValueError):
            ModelConfig(learning_rate=-0.1)  # Negative learning rate

        with pytest.raises(ValueError):
            ModelConfig(gamma=1.5)  # Gamma > 1

        with pytest.raises(ValueError):
            ModelConfig(batch_size=0)  # Zero batch size


class TestRiskConfig:
    """Test RiskConfig class"""

    def test_default_risk_config(self):
        """Test default risk configuration"""
        config = RiskConfig()
        assert config.max_position_size == 0.1
        assert config.max_drawdown == 0.2
        assert config.stop_loss == 0.05

    def test_risk_config_validation(self):
        """Test risk configuration validation"""
        with pytest.raises(ValueError):
            RiskConfig(max_position_size=1.5)  # > 1

        with pytest.raises(ValueError):
            RiskConfig(max_drawdown=0)  # <= 0

        with pytest.raises(ValueError):
            RiskConfig(max_leverage=0.5)  # < 1


class TestBaseModule:
    """Test BaseModule class"""

    def test_base_module_is_abstract(self):
        """Test that BaseModule cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseModule({})

    def test_concrete_module(self):
        """Test concrete module implementation"""

        class ConcreteModule(BaseModule):
            def _validate_config(self):
                pass

            def _initialize(self):
                self.initialized = True

            def process(self, data):
                return data * 2

        module = ConcreteModule({'test': 'value'})
        assert module.initialized is True
        assert module.process(5) == 10
        assert module.config == {'test': 'value'}


class TestExceptions:
    """Test custom exceptions"""

    def test_trading_exception(self):
        """Test TradingException"""
        exc = TradingException("Test error", context={'key': 'value'})
        assert exc.message == "Test error"
        assert exc.context == {'key': 'value'}
        assert exc.timestamp is not None

    def test_data_exception(self):
        """Test DataException"""
        with pytest.raises(DataException):
            raise DataException("Data error")

    def test_risk_exception(self):
        """Test RiskException"""
        exc = RiskException(
            "Risk violation",
            risk_type='max_drawdown',
            severity='critical'
        )
        assert exc.risk_type == 'max_drawdown'
        assert exc.severity == 'critical'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
