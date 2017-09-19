import numpy as np
import os

from keras.models import load_model
from src.soccer.game import Game
import tensorflow as tf
from src.models.trainer.soccer_train import architecture

# model = load_model('model-80.h5')
# input_shape = (1, 11, 9, 11)
input_shape = [1, 11, 9, 11]
game = Game()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # saver = tf.train.import_meta_graph('./models/tf/model.ckpt-3584.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('./models/tf/'))

    sess.run(init_op)
    board_tf1 = tf.to_float(tf.placeholder(dtype=tf.int64, shape=input_shape))
    board_tf2 = tf.to_float(tf.placeholder(dtype=tf.int64, shape=input_shape))
    boards = tf.concat([board_tf1, board_tf2], 0)

    # logits1 = architecture(boards, is_training=False)
    # predictions1 = tf.argmax(logits1, axis=-1)
    while True:
        board_first = game.boards[0].board.reshape(input_shape)
        board_second = game.boards[1].board.reshape(input_shape)
        print(board_first.shape)
        # action_first_player = model.predict(board_first)
        # action_second_player = model.predict(board_second)

        feed_dict = {
            boards: np.concatenate((board_first, board_second), 0)
        }

        # predictions2 = tf.argmax(board_tf2, axis=-1)
        # p1 = sess.run([predictions1], feed_dict=feed_dict)
        # print(p1)
        # print(p2)

        # print("First player suggested move: {}".format(np.argmax(action_first_player)))
        # print("Second player suggested move: {}".format(np.argmax(action_second_player)))
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
