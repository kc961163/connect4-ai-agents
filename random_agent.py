# random_agent.py

import random

class RandomAgent:
    """
    Agent that selects actions randomly.
    """

    def __init__(self):
        """
        Initializes the random agent.
        """
        pass

    def select_action(self, env):
        """
        Selects a random valid action.

        Args:
            env (ConnectFourEnv): The game environment.

        Returns:
            int: Selected action (column index).
        """
        valid_actions = env.get_valid_actions()
        return random.choice(valid_actions)