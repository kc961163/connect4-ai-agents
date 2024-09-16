# dqn_agent.py

import numpy as np
from tensorflow import keras
from collections import deque
import random

class DQNAgent:
    """
    Deep Q-Network Agent that learns to play Connect Four.
    """

    def __init__(self, state_size, action_size, name='DQNAgent', epsilon_decay=0.995):
        """
        Initializes the DQN agent.

        Args:
            state_size (int): Total number of cells in the game board.
            action_size (int): Total number of possible actions (columns).
            name (str): Name of the agent.
            epsilon_decay (float): Decay rate for epsilon.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.name = name

        # Hyperparameters
        self.memory = deque(maxlen=5000)  # Replay memory
        self.gamma = 0.95                 # Discount factor
        self.epsilon = 1.0                # Exploration rate
        self.epsilon_min = 0.01           # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.learning_rate = 0.001        # Learning rate for optimizer

        # Build the neural network model
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the neural network model.

        Returns:
            keras.Model: Compiled neural network model.
        """
        model = keras.Sequential([
            keras.Input(shape=(self.state_size,)),  # Explicit Input layer
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')  # Output layer for Q-values
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay memory.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, env):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            env (ConnectFourEnv): The game environment.

        Returns:
            int: Selected action (column index).
        """
        state = env.get_state().reshape(1, -1)  # Flatten the state
        valid_actions = env.get_valid_actions()

        if np.random.rand() <= self.epsilon:
            # Random action (exploration)
            return random.choice(valid_actions)

        # Predict Q-values for all actions
        act_values = self.model.predict(state, verbose=0)[0]

        # Mask invalid actions by setting their Q-values to negative infinity
        masked_q_values = [act_values[a] if a in valid_actions else -np.inf for a in range(self.action_size)]

        # Choose the action with the highest Q-value
        return np.argmax(masked_q_values)

    def replay(self, batch_size):
        """
        Trains the neural network using experiences sampled from replay memory.

        Args:
            batch_size (int): Number of experiences to sample.
        """
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)[0]

            if done:
                target[action] = reward  # If episode ends, the target is just the reward
            else:
                # Predict future reward
                t = self.model.predict(next_state, verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)

            # Reshape target to match the model's expected input
            target = target.reshape(1, -1)

            # Train the model
            self.model.fit(state, target, epochs=1, verbose=0)

        # Reduce exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay