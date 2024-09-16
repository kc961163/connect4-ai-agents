# train_agents.py

from connect_four_env import ConnectFourEnv
from dqn_agent import DQNAgent
from random_agent import RandomAgent
import numpy as np

# Training parameters
EPISODES = 800       # Total number of training episodes
BATCH_SIZE = 64       # Batch size for experience replay
RENDER = True         # Set to True to enable rendering

def train():
    """
    Trains the DQN agent by playing against the random agent.
    """
    # Initialize environment and agents
    env = ConnectFourEnv()
    state_size = env.ROWS * env.COLUMNS
    action_size = env.COLUMNS

    agent1 = DQNAgent(state_size, action_size, name='Agent1')  # The learning agent
    # In train_agents.py, after initializing agent1
    agent1.model.load_weights('agent1_model_episode_200.h5')
    # Calculate epsilon at episode 200
    agent1.epsilon = max(agent1.epsilon_min, agent1.epsilon_decay ** 200)
    agent2 = RandomAgent()                                     # The opponent

    for e in range(EPISODES):
        state = env.reset()
        state = state.reshape(1, -1)  # Flatten the state
        done = False

        # Render only specific episodes
        render_episode = RENDER and (e % 100 == 0)  # Render every 100 episodes if RENDER is True

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

            if render_episode:
                env.render()  # Render the board

            # Agent1 learns only from its own moves
            if env.current_player == -1:
                # Agent1 made the move
                if done and env.winner == -1:
                    reward = -1  # Negative reward if Agent1 loses
                agent1.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                # Train the agent after the episode ends
                if len(agent1.memory) > BATCH_SIZE:
                    agent1.replay(BATCH_SIZE)
                print(f"Episode {e+1}/{EPISODES}, Agent1 Reward: {reward}, Epsilon: {agent1.epsilon:.2f}")
                break

        # Save the model every 100 episodes
        if (e + 1) % 100 == 0:
            agent1.model.save(f"agent1_model_episode_{e+1}.h5")

if __name__ == "__main__":
    train()