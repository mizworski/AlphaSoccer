# from PaperSoccer.Agent import Agent
import os
import sys

parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent)

from src.soccer.board import Board


class Game:
    def __init__(self, player0_elo=4, player1_elo=4, length=10, width=8):
        # 1 board for both players
        # self.board = Board(length, width)

        # 2 boards for each player
        # self.board_player0 = Board(0, player0_elo, length, width)
        # self.board_player1 = Board(1, player1_elo, length, width)
        self.boards = [
            Board(0, player0_elo, length, width),
            Board(1, player1_elo, length, width)
        ]

        # self.agent0 = Agent(length, width)
        # self.agent1 = Agent(length, width)

    def run(self):
        # while self.board.
        print(42)

    def make_move(self, direction):
        second_player_action = (direction + 4) % 8
        self.boards[1].make_move(second_player_action)
        return self.boards[0].make_move(direction)
        # if player == 0:
        #     self.board_agent0.make_move(direction)
        #     return self.board_agent1.make_move(direction + 4 % 8)
        # else:
        #     self.board_agent0.make_move(direction + 4 % 8)
        #     return self.board_agent1.make_move(direction)


if __name__ == '__main__':
    g = Game()
    b = g.boards[0]
    # g.make_move(0)
    # g.make_move(4)
    print(g.make_move(0))
    print(g.make_move(5))

    # b.print_board()

    print(b.board.shape)
    b.print_layer(9)
