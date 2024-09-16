# connect_four_env.py

import numpy as np

class ConnectFourEnv:
    """
    Connect Four game environment.
    """

    def __init__(self, rows=4, columns=5):
        """
        Initializes the game environment with a board of specified size.

        Args:
            rows (int): Number of rows in the game board.
            columns (int): Number of columns in the game board.
        """
        self.ROWS = rows
        self.COLUMNS = columns
        self.action_space = [i for i in range(self.COLUMNS)]  # Valid action indices
        self.reset()  # Initialize the game board

    def reset(self):
        """
        Resets the game to the initial state.

        Returns:
            np.ndarray: The initial game board state.
        """
        self.board = np.zeros((self.ROWS, self.COLUMNS), dtype=int)  # Empty board
        self.game_over = False  # Indicates if the game has ended
        self.winner = None      # Stores the winner (1, -1, or 0 for draw)
        self.current_player = 1  # Player 1 starts; players are represented by 1 and -1
        return self.get_state()

    def get_state(self):
        """
        Returns a copy of the current game board state.

        Returns:
            np.ndarray: Current game board state.
        """
        return self.board.copy()

    def is_valid_action(self, action):
        """
        Checks if the action (column index) is valid (i.e., the column is not full).

        Args:
            action (int): The column index to check.

        Returns:
            bool: True if the action is valid, False otherwise.
        """
        return self.board[0][action] == 0

    def get_valid_actions(self):
        """
        Retrieves a list of valid actions (non-full columns).

        Returns:
            list: List of valid column indices.
        """
        return [col for col in range(self.COLUMNS) if self.is_valid_action(col)]

    def step(self, action):
        """
        Executes the action (drop a piece in the specified column) and updates the game state.

        Args:
            action (int): The column index where the piece is to be dropped.

        Returns:
            tuple:
                - np.ndarray: Next state of the game board.
                - int: Reward received after the action.
                - bool: True if the game is over, False otherwise.
                - dict: Additional information (empty in this case).
        """
        if not self.is_valid_action(action):
            # Invalid move results in a negative reward and ends the game
            return self.get_state(), -10, True, {'invalid_move': True}

        # Place the piece in the next available row in the specified column
        row = self.get_next_open_row(action)
        self.board[row][action] = self.current_player

        # Check for a win condition
        if self.check_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player
            reward = 1  # Positive reward for winning
        elif len(self.get_valid_actions()) == 0:
            # The board is full, and it's a draw
            self.game_over = True
            self.winner = 0  # No winner
            reward = 0  # Neutral reward for a draw
        else:
            # The game continues
            reward = 0  # No immediate reward
            self.current_player *= -1  # Switch to the other player

        return self.get_state(), reward, self.game_over, {}

    def get_next_open_row(self, col):
        """
        Finds the next open row in the specified column.

        Args:
            col (int): The column index to check.

        Returns:
            int: The row index where the piece can be placed.

        Raises:
            Exception: If the column is full.
        """
        for row in range(self.ROWS - 1, -1, -1):  # Start from the bottom row
            if self.board[row][col] == 0:
                return row
        raise Exception("Column is full")

    def check_win(self, piece):
        """
        Checks if the specified player has achieved a win condition.

        Args:
            piece (int): The player's piece (1 or -1).

        Returns:
            bool: True if the player has won, False otherwise.
        """
        # Horizontal check
        for row in range(self.ROWS):
            for col in range(self.COLUMNS - 3):
                if all(self.board[row][col + i] == piece for i in range(4)):
                    return True

        # Vertical check
        for col in range(self.COLUMNS):
            for row in range(self.ROWS - 3):
                if all(self.board[row + i][col] == piece for i in range(4)):
                    return True

        # Positive diagonal check (\)
        for row in range(self.ROWS - 3):
            for col in range(self.COLUMNS - 3):
                if all(self.board[row + i][col + i] == piece for i in range(4)):
                    return True

        # Negative diagonal check (/)
        for row in range(3, self.ROWS):
            for col in range(self.COLUMNS - 3):
                if all(self.board[row - i][col + i] == piece for i in range(4)):
                    return True

        return False

    def render(self):
        """
        Prints the current state of the game board to the console.
        """
        print(np.flip(self.board, 0))  # Flip the board vertically for display