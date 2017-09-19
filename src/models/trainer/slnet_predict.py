import tensorflow as tf
import os
import numpy as np
from src.soccer.game import Game

from src.models.trainer.soccer_train import get_estimator

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
# model_path = tf.train.latest_checkpoint(model_dir)
# saver = tf.train.import_meta_graph(model_path + '.meta')
estimator = get_estimator(run_config, params, model_dir)

input_shape = [1, 11, 9, 11]
N_ACTIONS = 8

game = Game()

train_input_fn = lambda x: tf.estimator.inputs.numpy_input_fn(
    x={"x": x},
    shuffle=False)

with tf.Session() as sess:
    # sess.run(tf.initialize_all_variables())
    # saver.restore(sess, model_path)
    # graph = tf.get_default_graph()

    # 'shuffle_batch:0'

    # logits1 = architecture(boards, reuse=None, is_training=False)
    # predictions1 = tf.argmax(logits1, axis=-1)
    while True:
        board_first = game.boards[0].board.reshape(input_shape)
        board_second = game.boards[1].board.reshape(input_shape)
        print(board_first.shape)
        x = np.concatenate((board_first, board_second), 0).astype(np.float32)
        all_logits = estimator.predict(input_fn=train_input_fn(x), predict_keys='probabilities')

        for logits in all_logits:
            print(logits)
        # logits = sess.run([logits], feed_dict=feed_dict)
        # logits_first = logits[0][0]
        # logits_second = logits[0][1]
        # print(logits_first)
        # print(logits_second)
        # moves_first = sorted(range(N_ACTIONS), key=lambda k: logits_first[k])
        # moves_seconds = sorted(range(N_ACTIONS), key=lambda k: logits_second[k])
        # print(moves_first)
        # print(moves_seconds)

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
