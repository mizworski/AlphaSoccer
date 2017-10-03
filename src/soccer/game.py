# from PaperSoccer.Agent import Agent
import os
import sys

parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent)

from src.soccer.board import Board


class Game:
    def __init__(self, length=10, width=8):
        self.boards = [
            Board(length, width),
            Board(length, width)
        ]

    def make_move(self, direction, verbosity=0):
        second_player_action = (direction + 4) % 8
        res = self.boards[0].make_move(direction)
        self.boards[1].make_move(second_player_action)

        if verbosity:
            self.boards[0].print_layer(8)
            self.boards[1].print_layer(8)

        return res


if __name__ == '__main__':
    g = Game()
    b = g.boards[0]
    print(g.make_move(0))
    print(g.make_move(5))

    print(b.board.shape)
    b.print_layer(9)
