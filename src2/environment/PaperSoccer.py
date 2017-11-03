from src2.environment.Board import Board
from gym.spaces import Box, Discrete


class Soccer:
    def __init__(self):
        self.board = None
        self.observation_space = [1, 11, 9, 12]
        # self.observation_space = Box(0, 1, shape=[11, 9, 12])
        self.action_space = Discrete(8)

    def get_legal_moves(self):
        return self.board.get_legal_moves()

    def step(self, action, verbose=0):
        reward, _ = self.board.make_move(action)
        if verbose:
            self.board.print_board()

        return self.board.board.reshape(self.observation_space), reward, reward != 0

    def reset(self, starting_game=0, verbose=0):
        self.board = Board(starting_game=starting_game)
        if verbose:
            self.board.print_board()

        return self.board.board.reshape(self.observation_space)
