"""Epsilon Greedy Exploration Strategy."""
import numpy as np
import torch
from abc import abstractmethod

class epsilon_greedy:
    """Abstract base class for observation functions."""

    def __init__(self):
        """Initialize observation function."""
        raise NotImplementedError("Subclasses must implement __init__ method")

    @abstractmethod
    def choose(self):
        """Subclasses must override this method."""
        raise NotImplementedError("Subclasses must implement choose method")

    @abstractmethod
    def reset(self):
        """Subclasses must override this method."""
        raise NotImplementedError("Subclasses must implement reset method")

class EpsilonGreedy(epsilon_greedy):
    """q-learning Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        """Initialize q-learning Epsilon Greedy Exploration Strategy."""
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space):
        """Choose action based on epsilon greedy strategy."""
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())
        else:
            action = np.argmax(q_table[state])

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        # print(self.epsilon)
        return action

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon

class dqn_epsilon_greedy(epsilon_greedy):
    """dqn Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon = 0.05, min_epsilon = 0.01, decay = 0.99) -> None:
        """Initialize dqn Epsilon Greedy Exploration Strategy"""
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        
    def choose(self, agent, state) -> int:
        """
        Choose action based on epsilon greedy strategy.
        agent: class: dqn_agent
        """
        if np.random.rand() < self.epsilon:
            action = int(agent.action_space.sample())
        else:
            action = agent.value_model(torch.FloatTensor(state).reshape(1, agent.observation_dims)).argmax().item()
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        # print(self.epsilon)
        return action
    
    def reset(self) -> None:
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon
