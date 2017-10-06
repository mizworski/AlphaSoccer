import tensorflow as tf
import os
import numpy as np
from src.soccer.game import Game

from src.models.supervised_policy.soccer_train import get_estimator

FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.DEBUG)

params = tf.contrib.training.HParams(
    learning_rate=0.001,
    min_eval_frequency=8,
    save_checkpoints_steps=8,
    num_epochs=FLAGS.num_epochs,
    batch_size=FLAGS.batch_size
)

run_config = tf.contrib.learn.RunConfig()

model_dir = os.path.join('models', 'tf')
estimator = get_estimator(run_config, params, model_dir)

input_shape = [1, 11, 9, 12]
N_ACTIONS = 8

game = Game()

train_input_fn = lambda x: tf.estimator.inputs.numpy_input_fn(
    x={"x": x},
    shuffle=False)

with tf.Session() as sess:
    while True:
        board_first = game.boards[0].board.reshape(input_shape)
        board_second = game.boards[1].board.reshape(input_shape)
        print(board_first.shape)
        x = np.concatenate((board_first, board_second), 0).astype(np.float32)
        all_logits = estimator.predict(input_fn=train_input_fn(x), predict_keys='probabilities')

        for logits in all_logits:
            moves = sorted(range(N_ACTIONS), key=lambda k: logits['probabilities'][k], reverse=True)
            print(moves)

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
