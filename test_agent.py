# test_agent.py

from connect_four_env import ConnectFourEnv
from dqn_agent import DQNAgent
from random_agent import RandomAgent
import numpy as np

def test():
    """
    Tests the trained DQN agent by playing against the random agent.
    """
    # Initialize environment and agents
    env = ConnectFourEnv()
    state_size = env.ROWS * env.COLUMNS
    action_size = env.COLUMNS

    agent1 = DQNAgent(state_size, action_size, name='Agent1')
    agent1.model.load_weights('agent1_model_episode_200.h5')  # Load the trained model
    agent1.epsilon = 0  # Disable exploration

    agent2 = RandomAgent()  # Opponent

    wins = 0
    losses = 0
    draws = 0
    test_episodes = 100  # Number of test games

    for e in range(test_episodes):
        state = env.reset()
        state = state.reshape(1, -1)
        done = False

        while not done:
            if env.current_player == 1:
                # Agent1's turn
                action = agent1.select_action(env)
            else:
                # Agent2's turn
                action = agent2.select_action(env)

            # Execute the action
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape(1, -1)

            state = next_state

            if done:
                if env.winner == 1:
                    wins += 1
                elif env.winner == -1:
                    losses += 1
                else:
                    draws += 1
                break

    # Display test results
    print(f"Out of {test_episodes} games:")
    print(f"Agent1 wins: {wins}")
    print(f"Agent1 losses: {losses}")
    print(f"Draws: {draws}")

if __name__ == "__main__":
    test()