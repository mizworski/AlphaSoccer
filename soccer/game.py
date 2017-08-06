# from PaperSoccer.Agent import Agent

from soccer.board import Board


class Game:
    def __init__(self, length=10, width=8):
        self.board = Board(length, width)
        # self.agent0 = Agent(length, width)
        # self.agent1 = Agent(length, width)

    def run(self):
        # while self.board.
        print(42)


if __name__ == '__main__':
    g = Game()
    b = g.board
    print(b.make_move(0))
    print(b.make_move(0))
    print(b.make_move(0))
    print(b.make_move(0))
    print(b.make_move(0))
    print(b.make_move(0))
    print(b.make_move(0))
    # print(b.make_move(0))
    # print(b.make_move(0))
    # print(b.make_move(0))
    # print(b.make_move(2))
    # print(b.make_move(0))
    # print(b.make_move(2))

    b.print_board()
