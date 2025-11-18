"""
Module: core.base
Description: Abstract base classes for the RL trading system
Author: Claude Code
Date: 2025-11-18
Version: 1.0.0

Dependencies:
    - abc: Abstract base class support
    - typing: Type hints
    - logging: Logging support

Usage Example:
    >>> from rl_trading_system.core.base import BaseModule
    >>> class MyModule(BaseModule):
    ...     def _validate_config(self):
    ...         pass
    ...     def _initialize(self):
    ...         pass
    ...     def process(self, data):
    ...         return data
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModule(ABC):
    """
    Abstract base class for all modules in the trading system

    All modules should inherit from this class and implement the required
    abstract methods. This ensures a consistent interface across the system.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the module

    Example:
        >>> class CustomModule(BaseModule):
        ...     def _validate_config(self) -> None:
        ...         if 'param1' not in self.config:
        ...             raise ValueError("Missing param1")
        ...
        ...     def _initialize(self) -> None:
        ...         self.param1 = self.config['param1']
        ...
        ...     def process(self, data):
        ...         return data * self.param1
        >>>
        >>> module = CustomModule({'param1': 2})
        >>> result = module.process(5)
        >>> print(result)  # Output: 10
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base module

        Args:
            config: Configuration dictionary containing module parameters

        Raises:
            ValueError: If configuration validation fails
        """
        self.config = config
        logger.info(f"Initializing {self.__class__.__name__}")
        self._validate_config()
        self._initialize()
        logger.info(f"{self.__class__.__name__} initialized successfully")

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate configuration parameters

        This method should check that all required configuration parameters
        are present and have valid values.

        Raises:
            ValueError: If any required parameter is missing or invalid
            TypeError: If parameter types are incorrect
        """
        pass

    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize module components

        This method should set up any internal state, load resources,
        or perform other initialization tasks after config validation.

        Raises:
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Main processing method

        This is the primary method that performs the module's core functionality.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Any: Processing result

        Raises:
            Exception: Various exceptions depending on implementation
        """
        pass

    def cleanup(self) -> None:
        """
        Cleanup resources

        This method should release any resources held by the module.
        Override this method if your module needs cleanup.
        """
        logger.info(f"Cleaning up {self.__class__.__name__}")
        pass

    def __repr__(self) -> str:
        """String representation of the module"""
        return f"{self.__class__.__name__}(config={self.config})"

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        return False


class BaseStrategy(BaseModule):
    """
    Abstract base class for trading strategies

    Strategies should inherit from this class and implement strategy-specific
    logic in the generate_signals method.

    Attributes:
        name (str): Strategy name
        parameters (Dict): Strategy parameters
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy

        Args:
            config: Strategy configuration including name and parameters
        """
        super().__init__(config)
        self.name = config.get('name', self.__class__.__name__)
        self.parameters = config.get('parameters', {})

    @abstractmethod
    def generate_signals(self, market_data: Any) -> Any:
        """
        Generate trading signals from market data

        Args:
            market_data: Market data to analyze

        Returns:
            Trading signals
        """
        pass

    def process(self, *args, **kwargs) -> Any:
        """
        Process method calls generate_signals

        This provides compatibility with BaseModule interface.
        """
        return self.generate_signals(*args, **kwargs)


class BaseAgent(BaseModule):
    """
    Abstract base class for reinforcement learning agents

    Agents should inherit from this class and implement the required methods
    for training and inference.

    Attributes:
        state_dim (int): State space dimension
        action_dim (int): Action space dimension
        learning_rate (float): Learning rate for training
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize agent

        Args:
            config: Agent configuration
        """
        super().__init__(config)
        self.state_dim = config.get('state_dim', 0)
        self.action_dim = config.get('action_dim', 0)
        self.learning_rate = config.get('learning_rate', 0.001)

    @abstractmethod
    def act(self, state: Any) -> Any:
        """
        Select action based on current state

        Args:
            state: Current environment state

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, float]:
        """
        Train the agent

        Returns:
            Training metrics dictionary
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent model to file

        Args:
            path: Path to save the model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent model from file

        Args:
            path: Path to load the model from
        """
        pass

    def process(self, *args, **kwargs) -> Any:
        """
        Process method calls act

        This provides compatibility with BaseModule interface.
        """
        return self.act(*args, **kwargs)
