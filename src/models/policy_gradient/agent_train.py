import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from src.soccer.PaperSoccer import Soccer
from src.models.networks.policy_network import get_policy_network

from collections import deque

tf.logging.set_verbosity(tf.logging.DEBUG)
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    flag_name='policies_dir', default_value='models/policy_networks/policy_gradient4/',
    docstring='Directory containing all policy networks.')

tf.app.flags.DEFINE_integer(
    flag_name='num_games', default_value='512',
    docstring='Number of games played between each policy iteration.'
)
tf.app.flags.DEFINE_integer(
    flag_name='sample_size', default_value='512',
    docstring='Size of learning sample from all games state-actions.'
)
tf.app.flags.DEFINE_integer(
    flag_name='history_size', default_value='2048',
    docstring='Number of state-actions to be stored in history.'
)
tf.app.flags.DEFINE_integer(
    flag_name='batch_size', default_value='256',
    docstring='Number of state-action pairs in single batch.'
)
tf.app.flags.DEFINE_integer(
    flag_name='print_board', default_value='64',
    docstring='Printing board every n games.'
)
tf.app.flags.DEFINE_integer(
    flag_name='num_iterations', default_value='2048',
    docstring='Number of policy iterations.'
)
tf.app.flags.DEFINE_integer(
    flag_name='max_turns_per_game', default_value='1024',
    docstring='Number of policy iterations.'
)
tf.app.flags.DEFINE_integer(
    flag_name='games_against_same_opponent', default_value='1',
    docstring='Number of games played before opponent changes.'
)
tf.app.flags.DEFINE_float(
    flag_name='initial_epsilon', default_value='0.01',
    docstring='Initial epsilon for epsilon-greedy exploration.'
)
tf.app.flags.DEFINE_float(
    flag_name='momentum', default_value='0.9',
    docstring='Momentum for SGD.'
)
tf.app.flags.DEFINE_integer(
    flag_name='last_k_models', default_value='5',
    docstring='Number of latest models to play against.'
)
tf.app.flags.DEFINE_integer(
    flag_name='training_steps', default_value='1',
    docstring='Training steps per single experience sample.'
)
tf.app.flags.DEFINE_float(
    flag_name='initial_learning_rate', default_value='1e-11',
    docstring='Initial learning rate.'
)

input_dim = [-1, 11, 9, 12]


def get_estimator(run_config, params, model_dir):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config,
        model_dir=model_dir
    )


def model_fn(features, labels, mode, params):
    rewards = labels
    logits = get_policy_network(features['states'])
    network_outputs = {
        'probabilities': tf.nn.softmax(logits),
        'labels': tf.argmax(logits, axis=-1)
    }

    loss = None
    train_op = None
    if mode != ModeKeys.INFER:
        logits = tf.Print(logits, [logits], message="logits:", summarize=8)
        probs = tf.nn.softmax(logits)
        probs = tf.Print(probs, [probs], message="probs:", summarize=8)
        log_probs = tf.log(probs)
        log_probs = tf.Print(log_probs, [log_probs], message="log_probs:", summarize=8)
        actions = features['actions']
        actions = tf.Print(actions, [actions], message="actions:", summarize=1)
        log_prob_indices = tf.range(0, 8 * tf.shape(log_probs)[0], delta=8) + actions
        log_prob_indices = tf.Print(log_prob_indices, [log_prob_indices], message="log_prob_indices:", summarize=1)
        log_prob_given_action = tf.gather(log_probs, log_prob_indices, axis=1)
        log_prob_given_action = tf.Print(log_prob_given_action, [log_prob_given_action],
                                         message="log_prob_given_action:", summarize=8)
        gain_single_action = tf.multiply(log_prob_given_action, rewards)
        gain_single_action = tf.Print(gain_single_action, [gain_single_action], message="gain_single_action:",
                                      summarize=1)
        loss = -tf.reduce_sum(gain_single_action)
        train_op = get_train_op_fn(loss, params)

    predictions = network_outputs if mode == ModeKeys.INFER else network_outputs['labels']
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )


def get_train_op_fn(loss, params):
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        optimizer=lambda lr: tf.train.MomentumOptimizer(lr, momentum=params.momentum),
        learning_rate=params.learning_rate
    )


def get_input_fn(states, actions, rewards):
    states = tf.reshape(np.asarray(states, dtype=np.float32), input_dim)
    actions = tf.reshape(np.asarray(actions, dtype=np.int32), [-1])
    rewards = tf.reshape(np.asarray(rewards, dtype=np.float32), [-1])
    sar_history = [states, actions, rewards]

    batch = tf.train.shuffle_batch(
        sar_history,
        FLAGS.batch_size,
        min_after_dequeue=2 * FLAGS.batch_size + 1,
        capacity=FLAGS.batch_size * 10,
        num_threads=multiprocessing.cpu_count(),
        enqueue_many=True,
        allow_smaller_final_batch=True,
    )

    features = {
        'states': batch[0],
        'actions': batch[1]
    }

    return features, batch[2]


def play_single_game(states, actions, rewards, env, verbose=0):
    inputs = graph.get_tensor_by_name("shuffle_batch:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    output = graph.get_tensor_by_name("PolicyNetwork/output/BiasAdd:0")

    starting_game = np.random.rand() < 0.5
    state = env.reset(starting_game=starting_game, verbose=0)

    for turn in range(FLAGS.max_turns_per_game):
        legal_moves = env.get_legal_moves()

        exploration_epsilon = FLAGS.initial_epsilon * (1 / (policy_iteration + 1) ** (1 / 2))
        if np.random.random() < exploration_epsilon:
            if sum(legal_moves) == 0:
                moves_prob = [0.125]*8
            else:
                moves_prob = [move / sum(legal_moves) for move in legal_moves]

            # action = int(np.random.random() * 8)
            action = np.random.choice(np.arange(8), p=moves_prob)
        else:
            feed_dict = {
                inputs: state,
                keep_prob: 1.0
            }

            logits = sess.run(output, feed_dict=feed_dict)
            acts = sorted(range(len(logits[0])), key=lambda k: logits[0][k], reverse=True)

            action = acts[0]
            for act in acts:
                if legal_moves[act] == 1:
                    action = act
                    break

            if verbose:
                env.player_board.print_board()
                print(legal_moves)
                print(action)
                print('Player')

        states.append(state[0])
        actions.append(action)
        state, reward, done = env.step(action, verbose=verbose)

        if done:
            rewards += [reward] * (turn + 1)
            # if reward == 1:
            #     env.player_board.print_board()
            #     print("Game won!")
            return max(min(reward, 1), 0)


def sample_from_experience(states, actions, rewards, sample_size):
    # todo prioriterized experience replay
    sample_size = sample_size if sample_size <= len(states) else len(states)
    idx = np.random.choice(len(states), sample_size, replace=False)
    states_sample = np.take(states, idx, axis=0)
    actions_sample = np.take(actions, idx)
    rewards_sample = np.take(rewards, idx)

    return states_sample, actions_sample, rewards_sample


if __name__ == '__main__':
    run_config = tf.contrib.learn.RunConfig()
    params = tf.contrib.training.HParams(
        learning_rate=FLAGS.initial_learning_rate,
        num_games=FLAGS.num_games,
        batch_size=FLAGS.batch_size,
        num_iterations=FLAGS.num_iterations,
        momentum=FLAGS.momentum
    )
    verbose = 0

    policy_network = get_estimator(run_config, params, FLAGS.policies_dir)

    states = deque(maxlen=FLAGS.history_size)
    actions = deque(maxlen=FLAGS.history_size)
    rewards = deque(maxlen=FLAGS.history_size)

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        for policy_iteration in range(FLAGS.num_iterations):
            env = Soccer(k_last_models=FLAGS.last_k_models)
            print("iteration={}".format(policy_iteration))

            model_path = tf.train.latest_checkpoint(FLAGS.policies_dir)
            saver = tf.train.import_meta_graph(model_path + '.meta')
            saver.restore(sess, model_path)

            for _ in range(FLAGS.games_against_same_opponent):
                games_won = 0
                for game in range(FLAGS.num_games):
                    outcome = play_single_game(states, actions, rewards, env, verbose)
                    games_won += outcome
                    if outcome > 1:
                        print(outcome)
                    if game % FLAGS.print_board == FLAGS.print_board - 1:
                        env.player_board.print_board()
                        print("Win ratio = {:.2f}%".format(100 * games_won / (game + 1)))

                states_sample, actions_sample, rewards_sample = sample_from_experience(states, actions, rewards,
                                                                                       FLAGS.sample_size)
                input_fn = lambda: get_input_fn(states_sample, actions_sample, rewards_sample)

                policy_network.train(input_fn, steps=FLAGS.training_steps)
