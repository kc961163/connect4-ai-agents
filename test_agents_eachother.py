from connect_four_env import ConnectFourEnv
from dqn_agent import DQNAgent
import numpy as np
import argparse
import os
import re

def find_latest_model(agent_name):
    """
    Finds the latest model file for the specified agent based on the highest episode number.

    Args:
        agent_name (str): The name of the agent (e.g., 'agent1', 'agent2', 'agent3').

    Returns:
        str: The path to the latest model file.
    """
    # Regular expression to match the model files
    pattern = re.compile(f"{agent_name}_model_episode_(\\d+).h5")
    latest_model = None
    latest_episode = -1

    # Search the current directory for model files
    for filename in os.listdir('.'):
        match = pattern.match(filename)
        if match:
            episode = int(match.group(1))
            if episode > latest_episode:
                latest_episode = episode
                latest_model = filename

    if latest_model is None:
        raise ValueError(f"No model file found for {agent_name}.")

    return latest_model

def test(agent_name1, agent_name2, episodes):
    """
    Tests two trained DQN agents by playing them against each other.

    Args:
        agent_name1 (str): Name of the first agent (e.g., 'agent1').
        agent_name2 (str): Name of the second agent (e.g., 'agent2').
        episodes (int): Number of episodes to test.
    """
    # Initialize environment
    env = ConnectFourEnv()
    state_size = env.ROWS * env.COLUMNS
    action_size = env.COLUMNS

    # Automatically find the latest model files for both agents
    model_path_agent1 = find_latest_model(agent_name1)
    model_path_agent2 = find_latest_model(agent_name2)

    # Print which model files are being used
    print(f"Using model for {agent_name1}: {model_path_agent1}")
    print(f"Using model for {agent_name2}: {model_path_agent2}")

    # Initialize the agents
    agent1 = DQNAgent(state_size, action_size, name=agent_name1.capitalize())
    agent1.model.load_weights(model_path_agent1)  # Load the trained model for Agent1
    agent1.epsilon = 0  # Disable exploration for testing

    agent2 = DQNAgent(state_size, action_size, name=agent_name2.capitalize())
    agent2.model.load_weights(model_path_agent2)  # Load the trained model for Agent2
    agent2.epsilon = 0  # Disable exploration for testing

    agent1_wins, agent2_wins, draws = 0, 0, 0
    total_moves = 0

    for e in range(episodes):
        state = env.reset()
        state = state.reshape(1, -1)
        done = False
        moves = 0

        while not done:
            if env.current_player == 1:
                # Agent1's turn
                action = agent1.select_action(env)
                current_agent = agent1
            else:
                # Agent2's turn
                action = agent2.select_action(env)
                current_agent = agent2

            # Execute the action
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape(1, -1)
            state = next_state
            moves += 1

            if done:
                total_moves += moves
                if env.winner == 1:
                    agent1_wins += 1
                elif env.winner == -1:
                    agent2_wins += 1
                else:
                    draws += 1
                break

    # Calculate results
    win_rate_agent1 = (agent1_wins / episodes) * 100
    win_rate_agent2 = (agent2_wins / episodes) * 100
    draw_rate = (draws / episodes) * 100
    avg_moves = total_moves / episodes

    # Display test results dynamically
    print(f"Testing completed over {episodes} games:")
    print(f"{agent_name1.capitalize()} wins: {agent1_wins} ({win_rate_agent1:.2f}%)")
    print(f"{agent_name2.capitalize()} wins: {agent2_wins} ({win_rate_agent2:.2f}%)")
    print(f"Draws: {draws} ({draw_rate:.2f}%)")
    print(f"Average moves per game: {avg_moves:.2f}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test two DQN agents against each other.')
    parser.add_argument('--agent1', type=str, default='agent1', help='Name of the first agent (e.g., agent1, agent2)')
    parser.add_argument('--agent2', type=str, default='agent2', help='Name of the second agent (e.g., agent2, agent3)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to test')
    
    args = parser.parse_args()

    # Run the test with specified arguments
    test(args.agent1, args.agent2, args.episodes)