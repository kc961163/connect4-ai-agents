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

    return latest_model, latest_episode

def train(agent_name1, agent_name2, episodes, batch_size, render):
    """
    Resumes training two DQN agents by playing them against each other.

    Args:
        agent_name1 (str): Name of the first agent (e.g., 'agent1').
        agent_name2 (str): Name of the second agent (e.g., 'agent2').
        episodes (int): Number of episodes to train.
        batch_size (int): Batch size for experience replay.
        render (bool): Whether to render the game.
    """
    # Initialize environment
    env = ConnectFourEnv()
    state_size = env.ROWS * env.COLUMNS
    action_size = env.COLUMNS

    # Find the latest model files and episode numbers for both agents
    model_path_agent1, resume_from1 = find_latest_model(agent_name1)
    model_path_agent2, resume_from2 = find_latest_model(agent_name2)

    print(f"Resuming training for {agent_name1} from: {model_path_agent1}")
    print(f"Resuming training for {agent_name2} from: {model_path_agent2}")

    # Initialize agents with their latest models
    agent1 = DQNAgent(state_size, action_size, name=agent_name1.capitalize())
    agent1.model.load_weights(model_path_agent1)
    agent1.epsilon = max(agent1.epsilon_min, agent1.epsilon_decay ** resume_from1)

    agent2 = DQNAgent(state_size, action_size, name=agent_name2.capitalize())
    agent2.model.load_weights(model_path_agent2)
    agent2.epsilon = max(agent2.epsilon_min, agent2.epsilon_decay ** resume_from2)

    # Determine the starting episode
    start_episode = max(resume_from1, resume_from2)

    # Continue training from the highest episode found
    for e in range(start_episode, episodes):
        state = env.reset()
        state = state.reshape(1, -1)
        done = False

        # Determine whether to render this episode
        render_episode = render and (e % 100 == 0)

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

            # Optionally render the environment
            if render_episode:
                env.render()

            # Store experience and train
            if done:
                if env.winner == env.current_player:
                    reward = 1  # Positive reward if the agent won
                elif env.winner == 0:
                    reward = 0  # Neutral reward for a draw
                else:
                    reward = -1  # Negative reward if the agent lost
                current_agent.remember(state, action, reward, next_state, done)
            else:
                reward = 0  # No reward until the game ends
                current_agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                # Train both agents
                if len(agent1.memory) > batch_size:
                    agent1.replay(batch_size)
                if len(agent2.memory) > batch_size:
                    agent2.replay(batch_size)

                # Print concise result of the episode
                if env.winner == 1:
                    print(f"Episode {e+1}/{episodes}: {agent_name1.capitalize()} wins! Epsilon1: {agent1.epsilon:.2f}, Epsilon2: {agent2.epsilon:.2f}")
                elif env.winner == -1:
                    print(f"Episode {e+1}/{episodes}: {agent_name2.capitalize()} wins! Epsilon1: {agent1.epsilon:.2f}, Epsilon2: {agent2.epsilon:.2f}")
                else:
                    print(f"Episode {e+1}/{episodes}: It's a draw! Epsilon1: {agent1.epsilon:.2f}, Epsilon2: {agent2.epsilon:.2f}")

                break

        # Save the models periodically
        if (e + 1) % 100 == 0:
            agent1.model.save(f"{agent_name1}_model_episode_{e+1}.h5")
            agent2.model.save(f"{agent_name2}_model_episode_{e+1}.h5")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train two DQN agents against each other.')
    parser.add_argument('--agent1', type=str, default='agent1', help='Name of the first agent (e.g., agent1, agent2)')
    parser.add_argument('--agent2', type=str, default='agent2', help='Name of the second agent (e.g., agent2, agent3)')
    parser.add_argument('--episodes', type=int, default=1000, help='Total number of training episodes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for experience replay')
    parser.add_argument('--render', action='store_true', help='Render the environment every 100 episodes')

    args = parser.parse_args()

    # Run the training with specified arguments
    train(args.agent1, args.agent2, args.episodes, args.batch_size, args.render)