import tensorflow as tf
import os
import numpy as np

from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner
import multiprocessing
from functools import reduce
import operator

from src.models.networks.policy_network import get_policy_network
from src.soccer.SoccerEnv import Soccer

tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string(
#     flag_name='initial_policy_dir', default_value='models/policy_networks/0_supervised/',
#     docstring='Directory containing all policy networks.')

tf.app.flags.DEFINE_string(
    flag_name='policies_dir', default_value='models/policy_networks/0_supervised/',
    docstring='Directory containing all policy networks.')

tf.app.flags.DEFINE_integer(
    flag_name='num_games', default_value='128',
    docstring='Number of games played between each policy iteration.'
)
tf.app.flags.DEFINE_integer(
    flag_name='batch_size', default_value='128',
    docstring='Number of state-action pairs in single batch.'
)
tf.app.flags.DEFINE_integer(
    flag_name='num_iterations', default_value='8',
    docstring='Number of policy iterations.'
)
tf.app.flags.DEFINE_integer(
    flag_name='max_turns_per_game', default_value='100',
    docstring='Number of policy iterations.'
)

params = tf.contrib.training.HParams(
    learning_rate=0.001,
    num_games=FLAGS.num_games,
    batch_size=FLAGS.batch_size,
    num_iterations=FLAGS.num_iterations
)

input_dim = [-1, 11, 9, 12]


def run_experiment(argv=None):
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.policies_dir)

    learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train",
        hparams=params
    )


if __name__ == '__main__':
    for itera\
            in range(FLAGS.num_iterations):
        env = Soccer(k_last_models=4)
        print(itera)

        model_path = tf.train.latest_checkpoint(FLAGS.policies_dir)
        saver = tf.train.import_meta_graph(model_path + '.meta')

        with tf.Session() as sess:
            saver.restore(sess, model_path)
            graph = tf.get_default_graph()

            inputs = graph.get_tensor_by_name("shuffle_batch:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            output = graph.get_tensor_by_name("PolicyNetwork/output/BiasAdd:0")

            for game in range(FLAGS.num_games):
                print(game)

                starting_game = np.random.rand() < 0.5
                state = env.reset(starting_game=starting_game, verbose=0)

                for _ in range(FLAGS.max_turns_per_game):

                    feed_dict = {
                        inputs: state,
                        keep_prob: 1.0
                    }

                    logits = sess.run([output], feed_dict=feed_dict)
                    action = np.argmax(logits)

                    state, reward, done = env.step(action, verbose=1)

                    if done:
                        print('Game ended, reward = {}'.format(reward))
                        break

                env.player_board.print_board()
