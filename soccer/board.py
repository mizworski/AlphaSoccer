import numpy as np


class Board:
    def __init__(self, length=10, width=8):
        assert length % 2 == 0 and width % 2 == 0

        self.length = length
        self.width = width
        self.ball_pos = (int(length / 2), int(width / 2))
        self.player_turn = 0
        self.board = np.zeros((8, length + 1, width + 1))

        for i in range(length + 1):
            for j in range(5, 8):
                self.board[j, i, 0] = 1
            for j in range(1, 4):
                self.board[j, i, width] = 1

        for i in range(width + 1):
            for j in {0, 1, 7}:
                self.board[j, 0, i] = 1
            for j in range(3, 6):
                self.board[j, length, i] = 1

        # player 0 scoring
        # possible to score from front
        self.board[0, 0, int(width / 2)] = 0
        self.board[1, 0, int(width / 2)] = 0
        self.board[7, 0, int(width / 2)] = 0

        # possible to score from sides
        self.board[1, 0, int(width / 2) - 1] = 0
        self.board[7, 0, int(width / 2) + 1] = 0

        # player 1 scoring
        # possible to score from front
        self.board[4, length, int(width / 2)] = 0
        self.board[5, length, int(width / 2)] = 0
        self.board[3, length, int(width / 2)] = 0

        # possible to score from sides
        self.board[5, length, int(width / 2) - 1] = 0
        self.board[3, length, int(width / 2) + 1] = 0

    def get_pos(self):
        return self.ball_pos

    def has_scored(self, direction):
        if self.ball_pos[0] == 0:
            if direction == 0 and self.ball_pos[1] == self.width / 2:
                return 0

            if direction == 1 and -1 + self.width / 2 <= self.ball_pos[1] <= self.width / 2:
                return 0

            if direction == 7 and self.width / 2 <= self.ball_pos[1] <= 1 + self.width / 2:
                return 0

        elif self.ball_pos[0] == self.length + 1:
            if direction == 4 and self.ball_pos[1] == self.width / 2:
                return 1

            if direction == 3 and -1 + self.width / 2 <= self.ball_pos[1] <= self.width / 2:
                return 1

            if direction == 5 and self.width / 2 <= self.ball_pos[1] <= 1 + self.width / 2:
                return 1

        return -1

    def out_of_board(self, x_delta, y_delta):
        if self.ball_pos[0] == 0 and x_delta <= 0:
            return False
        if self.ball_pos[1] == 0 and y_delta <= 0:
            return False
        if self.ball_pos[0] == self.length and x_delta >= 0:
            return False
        if self.ball_pos[1] == self.width and y_delta >= 0:
            return False

        return True

    def make_move(self, direction):
        x_delta = 0
        y_delta = 0

        game_winner = self.has_scored(direction)
        if game_winner == 0 or game_winner == 1:
            return 1

        if direction < 2 or direction > 6:
            x_delta = -1
        elif 2 < direction < 6:
            x_delta = 1

        if 0 < direction < 4:
            y_delta = 1
        elif 4 < direction < 8:
            y_delta = -1

        if not self.out_of_board(x_delta, y_delta):
            return -1

        if self.board[direction, self.ball_pos[0], self.ball_pos[1]] == 1:
            return -1

        self.board[direction, self.ball_pos[0], self.ball_pos[1]] = 1
        self.board[(direction + 4) % 8, self.ball_pos[0] + x_delta, self.ball_pos[1] + y_delta] = 1

        self.ball_pos = tuple(map(sum, zip(self.ball_pos, (x_delta, y_delta))))

        return 0

    def print_board(self):
        print(self.board)
