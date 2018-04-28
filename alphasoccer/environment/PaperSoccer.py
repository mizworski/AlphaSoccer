import numpy as np
from gym.spaces import Box, Discrete
from alphasoccer.environment.Board import Board

length=10
width=8

class Soccer:
    observation_space = Box(0, 1, shape=[length + 1, width + 1, 12])
    action_space = Discrete(8)

    def __init__(self):
        self.board = None

    def __str__(self):
        return str(self.board)

    def get_legal_moves(self):
        return self.board.get_legal_moves()

    def get_player_turn(self):
        return self.board.get_player_turn()

    def print_board(self):
        self.board.print_board()

    def step(self, action, verbose=0):
        reward = self.board.make_move(action)
        if verbose:
            self.print_board()

        return np.expand_dims(self.board.state, axis=0), reward, reward != 0

    def reset(self, starting_game=0, verbose=0):
        self.board = Board(starting_game=starting_game, length=length, width=width)
        if verbose:
            self.print_board()

        return np.expand_dims(self.board.state, axis=0)