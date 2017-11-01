from src2.environment.PaperSoccer import Soccer
from src2.policy_gradient.policy_network import get_policy_network

import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib import slim

# from src.soccer.PaperSoccer import Soccer
# from src.models.networks.policy_network import get_policy_network


from collections import deque

tf.logging.set_verbosity(tf.logging.DEBUG)
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    flag_name='policies_dir', default_value='models/policy_networks/policy_gradient/',
    docstring='Directory containing all policy networks.')

tf.app.flags.DEFINE_integer(
    flag_name='num_games', default_value='512',
    docstring='Number of games played between each policy iteration.'
)
tf.app.flags.DEFINE_integer(
    flag_name='sample_size', default_value='2048',
    docstring='Size of learning sample from all games state-actions.'
)
tf.app.flags.DEFINE_integer(
    flag_name='history_size', default_value='5000',
    docstring='Number of state-actions to be stored in history.'
)
tf.app.flags.DEFINE_integer(
    flag_name='batch_size', default_value='2048',
    docstring='Number of state-action pairs in single batch.'
)
tf.app.flags.DEFINE_integer(
    flag_name='print_board', default_value='256',
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
    flag_name='momentum', default_value='0.1',
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
    flag_name='initial_learning_rate', default_value='1e-8',
    docstring='Initial learning rate.'
)

INPUT_SHAPE = [-1, 11, 9, 12]
TURN_LAYER = 10


def get_estimator(run_config, params, model_dir):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config,
        model_dir=model_dir
    )


def get_train_op_fn(loss, params):
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        # optimizer=lambda lr: tf.train.MomentumOptimizer(lr, momentum=FLAGS.momentum),
        learning_rate=params.learning_rate
    )


def get_loss_reinforcement_learning(logits, rewards, actions):
    logits = tf.Print(logits, [logits], message="logits:", summarize=8)
    probs = tf.nn.softmax(logits)
    probs = tf.Print(probs, [probs], message="probs:", summarize=8)
    log_probs = tf.log(probs)
    log_probs = tf.Print(log_probs, [log_probs], message="log_probs:", summarize=8)
    actions = actions
    actions = tf.Print(actions, [actions], message="actions:", summarize=1)
    log_prob_indices = tf.range(0, 8 * tf.shape(log_probs)[0], delta=8) + actions
    log_prob_indices = tf.Print(log_prob_indices, [log_prob_indices], message="log_prob_indices:", summarize=1)
    log_prob_given_action = tf.gather(log_probs, log_prob_indices, axis=1)
    log_prob_given_action = tf.Print(log_prob_given_action, [log_prob_given_action],
                                     message="log_prob_given_action:", summarize=8)
    gain_single_action = tf.multiply(log_prob_given_action, rewards)
    gain_single_action = tf.Print(gain_single_action, [gain_single_action], message="gain_single_action:",
                                  summarize=1)
    loss = -tf.reduce_mean(gain_single_action)

    # reg_loss = tf.add_n(tf.losses.get_regularization_losses())
    # reg_loss = tf.Print(reg_loss, [reg_loss], message="reg_loss:", summarize=1)


    total_loss = loss
    # train_op = tf.train.MomentumOptimizer(FLAGS.initial_learning_rate, FLAGS.momentum).minimize(total_loss)

    return total_loss
    # return total_loss, train_op


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
        loss = get_loss_reinforcement_learning(logits, rewards, features['actions'])
        train_op = get_train_op_fn(loss, params)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=network_outputs,
        loss=loss,
        train_op=train_op
    )


def get_input_fn(states, actions, rewards):
    states = tf.reshape(np.asarray(states, dtype=np.float32), INPUT_SHAPE)
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


def play_single_game(graph, sess, history, env, opponent_graph, opponent_sess, env_opponent,
                     verbose=0):
    # todo rename tensors to friendly names
    inputs = graph.get_tensor_by_name("shuffle_batch:0")
    output = graph.get_tensor_by_name("SLNet/output/BiasAdd:0")
    inputs_opponent = opponent_graph.get_tensor_by_name("shuffle_batch:0")
    output_opponent = opponent_graph.get_tensor_by_name("SLNet/output/BiasAdd:0")
    player_turn = 0 if np.random.rand() < 0.5 else 1
    state = env.reset(verbose=verbose)
    state_opponent = env_opponent.reset(verbose=verbose)
    exploration_epsilon = FLAGS.initial_epsilon  # * (1 / (policy_iteration + 1) ** (1 / 2))

    states = []
    actions = []
    turn = 0
    while True:
        if player_turn == 0:
            state_new, reward, done, action = make_single_step(sess, inputs, output, env, state, exploration_epsilon,
                                                               verbose)
            action_opponent_perspective = (action + 4) % 8
            state_opponent, _, _ = env_opponent.step(action_opponent_perspective)
            states.append(state[0])
            actions.append(action)
            state = state_new
            turn += 1
        else:
            state_new_opp, _, _, action = make_single_step(opponent_sess, inputs_opponent, output_opponent,
                                                           env_opponent, state_opponent)
            action_player_perspective = (action + 4) % 8
            state, reward, done = env.step(action_player_perspective)
            state_opponent = state_new_opp
        player_turn = env.board.board[0, 0, TURN_LAYER]

        if done or turn > FLAGS.max_turns_per_game:
            if reward > 0:
                history['states_positive'] += states
                history['actions_positive'] += actions
            elif reward < 0:
                history['states_negative'] += states
                history['actions_negative'] += actions

            return reward if reward > 0 else 0


def make_single_step(sess, inputs, output, env, state, exploration_epsilon=0, verbose=0):
    legal_moves = env.get_legal_moves()
    if np.random.random() < exploration_epsilon:
        if sum(legal_moves) == 0:
            moves_prob = [0.125] * 8
        else:
            moves_prob = [move / sum(legal_moves) for move in legal_moves]

        action = np.random.choice(np.arange(8), p=moves_prob)
    else:
        network_output = sess.run(output, feed_dict={inputs: state})
        logits = network_output[0]

        acts = sorted(range(8), key=lambda k: logits[k], reverse=True)

        action = acts[0]
        for act in acts:
            if legal_moves[act] == 1:
                action = act
                break
    if verbose:
        env.board.print_board()
        print(legal_moves)
        print(action)
        print('Player')

    state_new, reward, done = env.step(action, verbose=verbose)
    return state_new, reward, done, action


def sample_from_experience(history, sample_size):
    history_len = min(len(history['states_positive']), len(history['states_negative']))
    sample_size = sample_size if sample_size <= history_len else history_len
    subsample_size = int(sample_size / 2)

    idx_p = np.random.choice(len(history['states_positive']), subsample_size, replace=False)
    idx_n = np.random.choice(len(history['states_negative']), subsample_size, replace=False)

    states_positive_sample = np.take(history['states_positive'], idx_p, axis=0)
    actions_positive_sample = np.take(history['actions_positive'], idx_p)
    rewards_positive_sample = [1] * subsample_size

    states_negative_sample = np.take(history['states_negative'], idx_n, axis=0)
    actions_negative_sample = np.take(history['actions_negative'], idx_n)
    rewards_negative_sample = [-1] * subsample_size

    # print(states_negative_sample.shape)
    # print(states_positive_sample.shape)
    states_sample = np.concatenate((states_positive_sample, states_negative_sample), axis=0)
    # print(states_sample.shape)
    actions_sample = np.concatenate((actions_positive_sample, actions_negative_sample))
    rewards_sample = np.concatenate((rewards_positive_sample, rewards_negative_sample))

    return states_sample, actions_sample, rewards_sample


def get_opponent_policy(k_last_models, policies_dir):
    checkpoint_state = tf.train.get_checkpoint_state(policies_dir)
    all_checkpoints = list(reversed(checkpoint_state.all_model_checkpoint_paths))
    k_last_models = k_last_models if k_last_models <= len(all_checkpoints) else len(all_checkpoints)

    chosen_model = int(k_last_models * np.random.random())

    model_path = all_checkpoints[chosen_model]
    print("Playing against {}".format(model_path))
    saver = tf.train.import_meta_graph(model_path + '.meta')

    sess = tf.Session()
    saver.restore(sess, model_path)
    opponent_policy_graph = tf.get_default_graph()

    return sess, opponent_policy_graph


def main():
    run_config = tf.contrib.learn.RunConfig()
    params = tf.contrib.training.HParams(
        learning_rate=FLAGS.initial_learning_rate,
        num_games=FLAGS.num_games,
        batch_size=FLAGS.batch_size,
        num_iterations=FLAGS.num_iterations
    )
    verbose = 0

    policy_network = get_estimator(run_config, params, FLAGS.policies_dir)

    history = {
        'states_positive': deque(maxlen=FLAGS.history_size),
        'states_negative': deque(maxlen=FLAGS.history_size),
        'actions_positive': deque(maxlen=FLAGS.history_size),
        'actions_negative': deque(maxlen=FLAGS.history_size)
    }
    # states = deque(maxlen=FLAGS.history_size)
    # actions = deque(maxlen=FLAGS.history_size)
    # rewards = deque(maxlen=FLAGS.history_size)

    env = Soccer()
    env_opponent = Soccer()

    k_last_models = 5

    with tf.Session() as sess:
        graph = tf.get_default_graph()

        model_path = tf.train.latest_checkpoint(FLAGS.policies_dir)
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)

        for policy_iteration in range(FLAGS.num_iterations):
            print("iteration={}".format(policy_iteration))
            opponent_sess, opponent_graph = get_opponent_policy(k_last_models, FLAGS.policies_dir)

            # if len(rewards) > 0:
            #     positive_rewards = 0
            #     negative_rewards = 0
            #     for r in rewards:
            #         if r == 1:
            #             positive_rewards += 1
            #         elif r == -1:
            #             negative_rewards += 1
            #
            #     print("Positive rewards {}".format(positive_rewards / len(rewards)) )
            #     print("Negative rewards {}".format(negative_rewards / len(rewards)) )

            for _ in range(FLAGS.games_against_same_opponent):
                games_won = 0
                for game in range(FLAGS.num_games):
                    outcome = play_single_game(graph, sess, history, env,
                                               opponent_graph, opponent_sess, env_opponent, verbose)
                    games_won += outcome
                    if outcome > 1:
                        print(outcome)
                    if game % FLAGS.print_board == FLAGS.print_board - 1:
                        env.board.print_board()
                        print("Win ratio = {:.2f}%".format(100 * games_won / (game + 1)))

                states_sample, actions_sample, rewards_sample = sample_from_experience(history,
                                                                                       FLAGS.sample_size)
                input_fn = lambda: get_input_fn(states_sample, actions_sample, rewards_sample)

                policy_network.train(input_fn, steps=FLAGS.training_steps)

                saver.save(sess, 'policy_network', global_step=policy_iteration)


if __name__ == '__main__':
    main()
