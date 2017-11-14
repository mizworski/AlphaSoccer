from src.environment.Board import Board
from gym.spaces import Box, Discrete


class Soccer:
    observation_space = Box(0, 1, shape=[11, 9, 12])
    action_space = Discrete(8)

    def __init__(self):
        self.board = None

    def get_legal_moves(self):
        return self.board.get_legal_moves()

    def get_player_turn(self):
        return self.board.get_player_turn()

    def print_board(self):
        self.board.print_board()

    def step(self, action, verbose=0):
        reward, _ = self.board.make_move(action)
        if verbose:
            self.board.print_board()

        return self.board.state.reshape((1,) + self.observation_space.shape), reward, reward != 0

    def reset(self, starting_game=0, verbose=0):
        self.board = Board(starting_game=starting_game)
        if verbose:
            self.board.print_board()

        return self.board.state.reshape((1,) + self.observation_space.shape)
