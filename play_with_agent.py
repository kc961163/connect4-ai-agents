# play_with_agent.py

from connect_four_env import ConnectFourEnv
from dqn_agent import DQNAgent
import numpy as np

def play():
    """
    Allows a human player to play against the trained DQN agent.
    """
    # Initialize environment and agent
    env = ConnectFourEnv()
    state_size = env.ROWS * env.COLUMNS
    action_size = env.COLUMNS

    agent = DQNAgent(state_size, action_size, name='Agent1')
    agent.model.load_weights('agent1_model_episode_200.h5')  # Load the trained model
    agent.epsilon = 0  # Disable exploration

    # Choose who goes first
    human_player = int(input("Do you want to go first? Enter 1 for Yes, 2 for No: "))
    if human_player == 1:
        human_player = 1
    else:
        human_player = -1

    env.render()

    done = False

    while not done:
        if env.current_player == human_player:
            # Human's turn
            valid_actions = env.get_valid_actions()
            action = None
            while action not in valid_actions:
                try:
                    action = int(input(f"Your turn! Choose a column {valid_actions}: "))
                    if action not in valid_actions:
                        print("Invalid action. Try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            _, _, done, _ = env.step(action)
        else:
            # Agent's turn
            action = agent.select_action(env)
            print(f"Agent chooses column {action}")
            _, _, done, _ = env.step(action)

        env.render()

        if done:
            if env.winner == human_player:
                print("You win!")
            elif env.winner == 0:
                print("It's a draw!")
            else:
                print("Agent wins!")
            break

if __name__ == "__main__":
    play()