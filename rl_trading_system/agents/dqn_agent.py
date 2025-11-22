"""
Module: agents.dqn_agent
Description: Deep Q-Network agent for trading
Author: Claude Code
Date: 2025-11-18
Version: 1.0.0

References:
    - Paper: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
    - Paper: "Quantitative Trading using Deep Q Learning" (arxiv:2304.06037)
    - Implementation: Stable-Baselines3

This module implements a DQN agent specifically designed for stock trading with:
- Experience replay
- Target network
- Epsilon-greedy exploration
- Double DQN (optional)
- Dueling DQN (optional)

Example:
    >>> from rl_trading_system.agents.dqn_agent import DQNAgent
    >>> agent = DQNAgent({
    ...     'state_dim': 20,
    ...     'action_dim': 3,
    ...     'learning_rate': 0.001
    ... })
    >>> action = agent.act(state)
    >>> agent.remember(state, action, reward, next_state, done)
    >>> loss = agent.train()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, Tuple, Optional, List, Any
import random
import logging

from rl_trading_system.core.base import BaseAgent
from rl_trading_system.core.exceptions import TrainingException, InferenceException
from rl_trading_system.utils.logger import get_logger

logger = get_logger(__name__)


class ReplayBuffer:
    """
    Experience replay buffer for DQN

    Stores transitions and samples random mini-batches for training.

    Attributes:
        capacity: Maximum buffer size
        buffer: Deque storing transitions
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add transition to buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample random batch

        Args:
            batch_size: Number of samples

        Returns:
            Batch of transitions
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.uint8)
        )

    def __len__(self) -> int:
        """Get buffer size"""
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture

    Standard fully-connected network for Q-value approximation.

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        hidden_dims: Hidden layer dimensions
        dueling: Whether to use dueling architecture
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        dueling: bool = False
    ):
        """Initialize DQN network"""
        super(DQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling

        # Build network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim

        if dueling:
            # Dueling DQN: separate value and advantage streams
            self.features = nn.Sequential(*layers)

            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
        else:
            # Standard DQN
            layers.append(nn.Linear(input_dim, action_dim))
            self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            state: State tensor

        Returns:
            Q-values for all actions
        """
        if self.dueling:
            features = self.features(state)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)

            # Combine value and advantage
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.network(state)

        return q_values


class DQNAgent(BaseAgent):
    """
    Deep Q-Network trading agent

    Implements DQN algorithm with experience replay and target network.

    Attributes:
        policy_net: Main Q-network
        target_net: Target Q-network
        optimizer: Network optimizer
        memory: Experience replay buffer
        epsilon: Current exploration rate

    Example:
        >>> config = {
        ...     'state_dim': 20,
        ...     'action_dim': 3,
        ...     'learning_rate': 0.001,
        ...     'gamma': 0.99,
        ...     'epsilon': 1.0,
        ...     'epsilon_decay': 0.995,
        ...     'epsilon_min': 0.01,
        ...     'batch_size': 32,
        ...     'memory_size': 10000
        ... }
        >>> agent = DQNAgent(config)
        >>> # Training loop
        >>> for episode in range(100):
        ...     state = env.reset()
        ...     while not done:
        ...         action = agent.act(state)
        ...         next_state, reward, done, _ = env.step(action)
        ...         agent.remember(state, action, reward, next_state, done)
        ...         state = next_state
        ...         if len(agent.memory) > agent.batch_size:
        ...             agent.train()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DQN agent

        Args:
            config: Configuration dictionary
                - state_dim (int): State dimension
                - action_dim (int): Action dimension
                - learning_rate (float): Learning rate
                - gamma (float): Discount factor
                - epsilon (float): Initial exploration rate
                - epsilon_decay (float): Epsilon decay rate
                - epsilon_min (float): Minimum epsilon
                - batch_size (int): Training batch size
                - memory_size (int): Replay buffer size
                - target_update_freq (int): Target network update frequency
                - hidden_dims (List[int]): Hidden layer dimensions
                - double_dqn (bool): Use double DQN
                - dueling_dqn (bool): Use dueling architecture
        """
        super().__init__(config)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"DQN using device: {self.device}")

        # Training parameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.batch_size = config.get('batch_size', 32)
        self.target_update_freq = config.get('target_update_freq', 100)
        self.double_dqn = config.get('double_dqn', False)

        # Network architecture
        hidden_dims = config.get('hidden_dims', [128, 128])
        dueling_dqn = config.get('dueling_dqn', False)

        # Initialize networks
        self.policy_net = DQNNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dims,
            dueling_dqn
        ).to(self.device)

        self.target_net = DQNNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dims,
            dueling_dqn
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )

        # Replay buffer
        memory_size = config.get('memory_size', 10000)
        self.memory = ReplayBuffer(memory_size)

        # Training step counter
        self.train_steps = 0

        logger.info(
            f"DQN initialized: state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"double={self.double_dqn}, dueling={dueling_dqn}"
        )

    def _validate_config(self) -> None:
        """Validate configuration"""
        required = ['state_dim', 'action_dim']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")

        if self.config['state_dim'] <= 0:
            raise ValueError("state_dim must be positive")

        if self.config['action_dim'] <= 0:
            raise ValueError("action_dim must be positive")

    def _initialize(self) -> None:
        """Initialize agent components"""
        # Already done in __init__
        pass

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        # Exploit: select best action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store transition in replay buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)

    def train(self) -> Dict[str, float]:
        """
        Train the agent on a batch from replay buffer

        Returns:
            Training metrics dictionary with loss
        """
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0}

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use policy net to select action, target net to evaluate
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_net(next_states).max(dim=1)[0]

        # Target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug(f"Target network updated at step {self.train_steps}")

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {'loss': loss.item(), 'epsilon': self.epsilon}

    def save(self, path: str) -> None:
        """
        Save model to file

        Args:
            path: File path to save model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps
        }, path)

        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load model from file

        Args:
            path: File path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.train_steps = checkpoint.get('train_steps', 0)

        logger.info(f"Model loaded from {path}")
