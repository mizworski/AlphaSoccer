import numpy as np

directions = 8
ball_layer = directions
dots_layer = ball_layer + 1
turn_layer = dots_layer + 1
bias_layer = turn_layer + 1


class Board:
    def __init__(self, starting_game=0, length=10, width=8):
        assert length % 2 == 0 and width % 2 == 0
        depth = bias_layer + 1

        self.length = length
        self.width = width

        # initialize board with zeros
        self.state = np.zeros((length + 1, width + 1, depth), dtype=np.float32)
        self.state[:, :, bias_layer] = 1

        # initial ball position in middle
        self.ball_pos = (int(length / 2), int(width / 2))
        self.state[self.ball_pos[0], self.ball_pos[1], ball_layer] = 1
        self.state[self.ball_pos[0], self.ball_pos[1], dots_layer] = 1

        # player 0 starts game
        # self.player_turn = starting_game
        self.state[:, :, turn_layer] = starting_game

        # 1 padding map
        for i in range(length + 1):
            # can't go (North, South)-West when you are on left edge
            for j in range(5, 8):
                self.state[i, 0, j] = 1

            # can't go (North, South)-East when you are on right edge
            for j in range(1, 4):
                self.state[i, width, j] = 1

            # you can bounce off edges even if ball wasn't there yet
            self.state[i, 0, dots_layer] = 1
            self.state[i, width, dots_layer] = 1

        for i in range(width + 1):
            # can't go North-(East, West)
            for j in {0, 1, 7}:
                self.state[0, i, j] = 1
            # can't go South-(East, West)
            for j in range(3, 6):
                self.state[length, i, j] = 1

            # bouncing off edge
            self.state[0, i, dots_layer] = 1
            self.state[length, i, dots_layer] = 1

        # todo is below necessary?
        for i in range(width + 1):
            self.state[0, i, 2] = 1
            self.state[0, i, 6] = 1
            self.state[length, i, 2] = 1
            self.state[length, i, 6] = 1

        for i in range(length + 1):
            self.state[i, 0, 0] = 1
            self.state[i, 0, 4] = 1
            self.state[i, width, 0] = 1
            self.state[i, width, 4] = 1

        # you can't bounce off point in front of goals
        self.state[0, int(self.width / 2), dots_layer] = 0
        self.state[self.length, int(self.width / 2), dots_layer] = 0

        # player 0 scoring
        # possible to score from front
        self.state[0, int(width / 2), 0] = 0
        self.state[0, int(width / 2), 1] = 0
        self.state[0, int(width / 2), 2] = 0
        self.state[0, int(width / 2), 6] = 0
        self.state[0, int(width / 2), 7] = 0

        # possible to score from sides
        self.state[0, int(width / 2) - 1, 1] = 0
        self.state[0, int(width / 2) - 1, 2] = 0
        self.state[0, int(width / 2) + 1, 6] = 0
        self.state[0, int(width / 2) + 1, 7] = 0

        # player 1 scoring
        # possible to score from front
        self.state[length, int(width / 2), 2] = 0
        self.state[length, int(width / 2), 3] = 0
        self.state[length, int(width / 2), 4] = 0
        self.state[length, int(width / 2), 5] = 0
        self.state[length, int(width / 2), 6] = 0

        # possible to score from sides
        self.state[length, int(width / 2) - 1, 2] = 0
        self.state[length, int(width / 2) - 1, 3] = 0
        self.state[length, int(width / 2) + 1, 5] = 0
        self.state[length, int(width / 2) + 1, 6] = 0

    def get_pos(self):
        return self.ball_pos

    def has_scored(self, direction):
        if self.ball_pos[0] == 0:
            if direction == 0 and self.ball_pos[1] == self.width / 2:
                return 1

            if direction == 1 and -1 + self.width / 2 <= self.ball_pos[1] <= self.width / 2:
                return 1

            if direction == 7 and self.width / 2 <= self.ball_pos[1] <= 1 + self.width / 2:
                return 1

        elif self.ball_pos[0] == self.length:
            if direction == 4 and self.ball_pos[1] == self.width / 2:
                return -1

            if direction == 3 and -1 + self.width / 2 <= self.ball_pos[1] <= self.width / 2:
                return -1

            if direction == 5 and self.width / 2 <= self.ball_pos[1] <= 1 + self.width / 2:
                return -1

        return 0

    def make_move(self, direction):
        x_delta = 0
        y_delta = 0
        player_taking_action = self.state[0, 0, turn_layer]

        game_winner = self.has_scored(direction)
        if game_winner != 0:
            return game_winner, False

        if direction < 2 or direction > 6:
            x_delta = -1
        elif 2 < direction < 6:
            x_delta = 1

        if 0 < direction < 4:
            y_delta = 1
        elif 4 < direction < 8:
            y_delta = -1

        if self.state[self.ball_pos[0], self.ball_pos[1], direction] == 1:
            return -1, True

        self.state[self.ball_pos[0], self.ball_pos[1], direction] = 1
        self.state[self.ball_pos[0] + x_delta, self.ball_pos[1] + y_delta, (direction + 4) % 8] = 1

        self.state[self.ball_pos[0], self.ball_pos[1], ball_layer] = 0
        self.ball_pos = tuple(map(sum, zip(self.ball_pos, (x_delta, y_delta))))
        self.state[self.ball_pos[0], self.ball_pos[1], ball_layer] = 1
        if self.state[self.ball_pos[0], self.ball_pos[1], dots_layer] == 0:
            self.state[self.ball_pos[0], self.ball_pos[1], dots_layer] = 1
            self.state[:, :, turn_layer] = 1 - player_taking_action

        return 0, player_taking_action == self.state[0, 0, turn_layer]

    def print_layer(self, k):
        layer = np.dsplit(self.state, self.state.shape[2])[k].reshape((self.state.shape[0], self.state.shape[1]))
        print(layer)

    def print_board(self):
        for row in range(int(self.length - 4)):
            print(' ', end='')
        print('+-+-+')

        for row in range(self.length + 1):
            for col in range(self.width + 1):
                if self.state[row, col, ball_layer] == 1:
                    print('O', end='')
                elif self.state[row, col, dots_layer] == 1:
                    print('+', end='')
                else:
                    print('.', end='')

                if col != self.width and (self.state[row, col, 2] == 1):
                    print('-', end='')
                elif col != self.width:
                    print(' ', end='')
            print('')

            if row != self.length:
                for col in range(self.width + 1):
                    if self.state[row, col, 4] == 1:
                        print('|', end='')
                    elif col != self.width:
                        print(' ', end='')

                    if col != self.width and self.state[row, col, 3] == 1:
                        if self.state[row + 1, col, 1] == 1:
                            print('X', end='')
                        else:
                            print('\\', end='')
                    elif col != self.width and self.state[row + 1, col, 1] == 1:
                        print('/', end='')
                    elif col != self.width:
                        print(' ', end='')

                print('')

        for row in range(int(self.length - 4)):
            print(' ', end='')
        print('+-+-+')

    def get_legal_moves(self):
        ball_pos = self.get_pos()
        deep_column = self.state[ball_pos]

        return [1 - deep_column[k] for k in range(8)]
