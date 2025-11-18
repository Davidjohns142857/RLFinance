#!/usr/bin/env python
"""
Comprehensive test script for RL Trading System

This script tests all core functionality without requiring external dependencies
like pytest or pandas/numpy (for basic tests).

Usage:
    python run_tests.py
"""

import sys
import os
import tempfile
import time
import json

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all core modules can be imported"""
    print("\n" + "=" * 50)
    print("Testing Imports")
    print("=" * 50)

    try:
        from rl_trading_system.core import (
            TradingConfig,
            BaseModule,
            BaseStrategy,
            BaseAgent,
            TradingException
        )
        print("✓ Core modules imported successfully")

        from rl_trading_system.utils import (
            setup_logging,
            get_logger,
            LogContext,
            PerformanceLogger
        )
        print("✓ Utils modules imported successfully")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration management"""
    print("\n" + "=" * 50)
    print("Testing Configuration")
    print("=" * 50)

    from rl_trading_system.core import TradingConfig

    try:
        # Test default config
        config = TradingConfig()
        assert config.data.source == "binance"
        assert config.model.learning_rate == 0.001
        print("✓ Default configuration created")

        # Test modification
        config.data.symbols = ['BTC/USDT', 'ETH/USDT']
        config.model.learning_rate = 0.0005
        assert len(config.data.symbols) == 2
        print("✓ Configuration modified")

        # Test save/load
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        config.save_to_file(temp_file)
        loaded = TradingConfig.load_from_file(temp_file)
        assert loaded.data.symbols == config.data.symbols
        os.unlink(temp_file)
        print("✓ Configuration save/load works")

        # Test validation
        config.validate()
        print("✓ Configuration validation passed")

        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exceptions():
    """Test exception handling"""
    print("\n" + "=" * 50)
    print("Testing Exceptions")
    print("=" * 50)

    from rl_trading_system.core.exceptions import (
        TradingException,
        DataException,
        RiskException,
        DrawdownException
    )

    try:
        # Test basic exception
        try:
            raise TradingException("Test error", context={'key': 'value'})
        except TradingException as e:
            assert e.message == "Test error"
            assert e.context == {'key': 'value'}
        print("✓ TradingException works")

        # Test RiskException
        try:
            raise RiskException(
                "Risk violation",
                risk_type='max_drawdown',
                severity='high'
            )
        except RiskException as e:
            assert e.risk_type == 'max_drawdown'
            assert e.severity == 'high'
        print("✓ RiskException works")

        # Test DrawdownException
        try:
            raise DrawdownException(
                "Drawdown exceeded",
                current_drawdown=0.25,
                max_drawdown=0.20
            )
        except DrawdownException as e:
            assert e.risk_metrics['current_drawdown'] == 0.25
        print("✓ DrawdownException works")

        return True
    except Exception as e:
        print(f"✗ Exception test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_classes():
    """Test base classes"""
    print("\n" + "=" * 50)
    print("Testing Base Classes")
    print("=" * 50)

    from rl_trading_system.core.base import BaseModule, BaseStrategy, BaseAgent

    try:
        # Test BaseModule implementation
        class TestModule(BaseModule):
            def _validate_config(self):
                pass

            def _initialize(self):
                self.initialized = True

            def process(self, data):
                return data * 2

        module = TestModule({'test': 'value'})
        assert module.initialized
        assert module.process(5) == 10
        print("✓ BaseModule works")

        # Test BaseStrategy implementation
        class TestStrategy(BaseStrategy):
            def _validate_config(self):
                pass

            def _initialize(self):
                pass

            def generate_signals(self, market_data):
                return {'signal': 'buy'}

        strategy = TestStrategy({'name': 'test'})
        assert strategy.name == 'test'
        assert strategy.generate_signals({})['signal'] == 'buy'
        print("✓ BaseStrategy works")

        # Test BaseAgent implementation
        class TestAgent(BaseAgent):
            def _validate_config(self):
                pass

            def _initialize(self):
                pass

            def act(self, state):
                return 0

            def train(self):
                return {'loss': 0.1}

            def save(self, path):
                pass

            def load(self, path):
                pass

        agent = TestAgent({'state_dim': 10, 'action_dim': 3})
        assert agent.state_dim == 10
        assert agent.action_dim == 3
        assert agent.act('state') == 0
        print("✓ BaseAgent works")

        return True
    except Exception as e:
        print(f"✗ Base classes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logging():
    """Test logging functionality"""
    print("\n" + "=" * 50)
    print("Testing Logging")
    print("=" * 50)

    from rl_trading_system.utils import (
        setup_logging,
        get_logger,
        LogContext,
        PerformanceLogger
    )

    temp_dir = tempfile.mkdtemp()

    try:
        # Setup logging
        setup_logging(
            level='INFO',
            log_dir=temp_dir,
            log_to_console=False
        )
        logger = get_logger('test')
        logger.info('Test message')
        print("✓ Logging setup works")

        # Test LogContext
        with LogContext(trade_id='12345'):
            logger.info('Context message')
        print("✓ LogContext works")

        # Test PerformanceLogger
        perf = PerformanceLogger('test_op', logger)
        with perf:
            time.sleep(0.001)
        print("✓ PerformanceLogger works")

        # Check log files
        log_files = os.listdir(temp_dir)
        assert len(log_files) > 0
        print(f"✓ Created {len(log_files)} log files")

        return True
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validation():
    """Test configuration validation"""
    print("\n" + "=" * 50)
    print("Testing Validation")
    print("=" * 50)

    from rl_trading_system.core.config import (
        DataConfig,
        ModelConfig,
        RiskConfig
    )

    try:
        # Test valid configs
        DataConfig(symbols=['BTC/USDT'])
        ModelConfig(learning_rate=0.001)
        RiskConfig(max_position_size=0.1)
        print("✓ Valid configurations accepted")

        # Test invalid configs
        validation_errors = 0

        try:
            DataConfig(symbols=[])
        except ValueError:
            validation_errors += 1

        try:
            ModelConfig(learning_rate=-0.1)
        except ValueError:
            validation_errors += 1

        try:
            RiskConfig(max_position_size=1.5)
        except ValueError:
            validation_errors += 1

        assert validation_errors == 3
        print("✓ Invalid configurations rejected")

        return True
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" " * 15 + "RL Trading System - Test Suite")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Exceptions", test_exceptions),
        ("Base Classes", test_base_classes),
        ("Logging", test_logging),
        ("Validation", test_validation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<50} {status}")

    print("=" * 70)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
