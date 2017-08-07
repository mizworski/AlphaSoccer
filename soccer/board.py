import numpy as np

directions = 8
ball_layer = directions
dots_layer = ball_layer + 1
player_color_layer = dots_layer + 1
# nearby_dots_layer = player_color_layer + 1
elo_layer = player_color_layer + 1


# todo moves from position possible

class Board:
    def __init__(self, player_color, player_elo, length=10, width=8):
        assert length % 2 == 0 and width % 2 == 0
        depth = elo_layer + 5  # number of layers before elo layers + number of elo layers
        player_elo_group = self.convert_elo(player_elo)

        self.length = length
        self.width = width
        self.ball_pos = (int(length / 2), int(width / 2))
        self.player_turn = 0
        self.board = np.zeros((length + 1, width + 1, depth))

        self.board[self.ball_pos[0], self.ball_pos[1], ball_layer] = 1
        self.board[self.ball_pos[0], self.ball_pos[1], dots_layer] = 1

        for i in range(length + 1):
            for j in range(width + 1):
                self.board[i, j, elo_layer + player_elo_group] = 1

        if player_color == 1:
            for i in range(length + 1):
                for j in range(width + 1):
                    self.board[i, j, player_color_layer] = 1

        for i in range(length + 1):
            for j in range(5, 8):
                self.board[i, 0, j] = 1
            for j in range(1, 4):
                self.board[i, width, j] = 1

            self.board[i, 0, dots_layer] = 1
            self.board[i, width, dots_layer] = 1

        for i in range(width + 1):
            for j in {0, 1, 7}:
                self.board[0, i, j] = 1
            for j in range(3, 6):
                self.board[length, i, j] = 1

            self.board[0, i, dots_layer] = 1
            self.board[length, i, dots_layer] = 1

        # player 0 scoring
        # possible to score from front
        self.board[0, int(width / 2), 0] = 0
        self.board[0, int(width / 2), 1] = 0
        self.board[0, int(width / 2), 7] = 0

        # possible to score from sides
        self.board[0, int(width / 2) - 1, 1] = 0
        self.board[0, int(width / 2) + 1, 7] = 0

        # player 1 scoring
        # possible to score from front
        self.board[length, int(width / 2), 4] = 0
        self.board[length, int(width / 2), 5] = 0
        self.board[length, int(width / 2), 3] = 0

        # possible to score from sides
        self.board[length, int(width / 2) - 1, 3] = 0
        self.board[length, int(width / 2) + 1, 5] = 0

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
        if game_winner >= 0:
            return game_winner

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

        if self.board[self.ball_pos[0], self.ball_pos[1], direction] == 1:
            return -1

        self.board[self.ball_pos[0], self.ball_pos[1], direction] = 1
        self.board[self.ball_pos[0] + x_delta, self.ball_pos[1] + y_delta, (direction + 4) % 8] = 1

        self.board[self.ball_pos[0], self.ball_pos[1], ball_layer] = 0
        self.ball_pos = tuple(map(sum, zip(self.ball_pos, (x_delta, y_delta))))
        self.board[self.ball_pos[0], self.ball_pos[1], ball_layer] = 1
        self.board[self.ball_pos[0], self.ball_pos[1], dots_layer] = 1

        return 2

    def print_layer(self, k):
        layer = np.dsplit(self.board, self.board.shape[2])[k].reshape((self.board.shape[0], self.board.shape[1]))
        print(layer)

    @staticmethod
    def convert_elo(elo):
        if elo < 1150:
            return 0
        if elo < 1250:
            return 1
        if elo < 1400:
            return 2
        if elo < 1650:
            return 3

        return 4
