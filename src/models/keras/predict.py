import numpy as np
import os

from keras.models import load_model
from src.soccer.game import Game

model = load_model('model-80.h5')
input_shape = (1, 11, 9, 11)
game = Game()
while True:
    board_first = game.boards[0].board.reshape(input_shape)
    board_second = game.boards[1].board.reshape(input_shape)
    print(board_first.shape)
    action_first_player = model.predict(board_first)
    action_second_player = model.predict(board_second)
    print("First player suggested move: {}".format(np.argmax(action_first_player)))
    print("Second player suggested move: {}".format(np.argmax(action_second_player)))
    print("Input next move that was made.")
    action = input()
    action = int(action)
    if action not in range(8):
        print("not valid action")
    elif action == -1:
        break
    else:
        res = game.make_move(action)
        print('Game returned {}'.format(res))
