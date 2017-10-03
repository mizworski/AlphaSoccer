import numpy as np
import os

from keras.models import load_model
from src.soccer.game import Game
import tensorflow as tf
from src.models.trainer.soccer_train import architecture

input_shape = [1, 11, 9, 12]
N_ACTIONS = 8

game = Game()

model_dir = os.path.join('models', 'tf')
model_path = tf.train.latest_checkpoint(model_dir)
saver = tf.train.import_meta_graph(model_path + '.meta')

with tf.Session() as sess:
    saver.restore(sess, model_path)
    graph = tf.get_default_graph()

    while True:
        board_first = game.boards[0].board.reshape(input_shape)
        board_second = game.boards[1].board.reshape(input_shape)
        print(board_first.shape)

        batch = graph.get_tensor_by_name("shuffle_batch:0")
        logits = graph.get_tensor_by_name("SLNet/output/BiasAdd:0")
        feed_dict = {
            batch: np.concatenate((board_first, board_second), 0)
        }

        logits = sess.run([logits], feed_dict=feed_dict)
        logits_first = logits[0][0]
        logits_second = logits[0][1]
        print(logits_first)
        print(logits_second)
        moves_first = sorted(range(N_ACTIONS), key=lambda k: logits_first[k], reverse=True)
        moves_seconds = sorted(range(N_ACTIONS), key=lambda k: logits_second[k], reverse=True)
        print(moves_first)
        print(moves_seconds)

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
