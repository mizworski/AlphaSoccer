class Board:
    def __init__(self, length=10, width=8):
        assert length % 2 == 0 and width % 2 == 0

        self.length = length
        self.width = width
        self.ball_pos = (length / 2, width / 2)
        self.player_turn = 0


    def get_pos(self):
        return self.ball_pos
