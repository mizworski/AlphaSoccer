import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner

import multiprocessing
from src.soccer.SoccerEnv import Soccer

from src.models.networks.policy_network import get_policy_network

tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string(
#     flag_name='initial_policy_dir', default_value='models/policy_networks/0_supervised/',
#     docstring='Directory containing all policy networks.')

tf.app.flags.DEFINE_string(
    flag_name='policies_dir', default_value='models/policy_networks/policy_gradient/',
    docstring='Directory containing all policy networks.')

tf.app.flags.DEFINE_integer(
    flag_name='num_games', default_value='2048',
    docstring='Number of games played between each policy iteration.'
)
tf.app.flags.DEFINE_integer(
    flag_name='sample_size', default_value='4096',
    docstring='Size of learning sample from all games state-actions.'
)
tf.app.flags.DEFINE_integer(
    flag_name='batch_size', default_value='1024',
    docstring='Number of state-action pairs in single batch.'
)
tf.app.flags.DEFINE_integer(
    flag_name='num_iterations', default_value='2048',
    docstring='Number of policy iterations.'
)
tf.app.flags.DEFINE_integer(
    flag_name='max_turns_per_game', default_value='250',
    docstring='Number of policy iterations.'
)
tf.app.flags.DEFINE_integer(
    flag_name='games_against_same_opponent', default_value='1',
    docstring='Number of games played before opponent changes.'
)
tf.app.flags.DEFINE_float(
    flag_name='exploratory_epsilon', default_value='0.005',
    docstring='Epsilon for epsilon-greedy exploration.'
)
tf.app.flags.DEFINE_integer(
    flag_name='print_board', default_value='128',
    docstring='Printing board every n games.'
)

params = tf.contrib.training.HParams(
    learning_rate=1e-8,
    num_games=FLAGS.num_games,
    batch_size=FLAGS.batch_size,
    num_iterations=FLAGS.num_iterations
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
    predictions = {
        'probabilities': tf.nn.softmax(logits),
        'labels': tf.argmax(logits, axis=-1)
    }

    loss = None
    train_op = None
    if mode != ModeKeys.INFER:
        logits = tf.Print(logits, [logits], message="logits:", summarize=24)

        probs = tf.nn.softmax(logits)
        probs = tf.Print(probs, [probs], message="probs:", summarize=24)

        log_prob = tf.log(probs)
        log_prob = tf.Print(log_prob, [log_prob], message="log_prob:", summarize=24)

        actions = features['actions']
        actions = tf.Print(actions, [actions], message="actions:", summarize=3)

        indices = tf.range(0, 8 * tf.shape(log_prob)[0], delta=8) + actions
        indices = tf.Print(indices, [indices], message="indices:", summarize=3)

        grad_act = tf.gather(probs, indices, axis=1)
        grad_act = tf.Print(grad_act, [grad_act], message="grad_act:", summarize=24)

        real_grad = tf.multiply(grad_act, rewards)
        real_grad = tf.Print(real_grad, [real_grad], message="real_grad:", summarize=2)

        loss = tf.reduce_sum(real_grad)

        train_op = get_train_op_fn(loss, params)

    predictions2 = predictions if mode == ModeKeys.INFER else predictions['labels']
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions2,
        loss=loss,
        train_op=train_op
    )


def get_train_op_fn(loss, params):
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.learning_rate
    )


def get_input_fn(states, actions, rewards):
    states = tf.reshape(np.asarray(states, dtype=np.float32), input_dim)
    actions = tf.reshape(np.asarray(actions, dtype=np.int32), [-1])
    # actions = tf.reshape(np.asarray(actions, dtype=np.int32), [-1, 8])
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


if __name__ == '__main__':
    run_config = tf.contrib.learn.RunConfig()
    policy_network = get_estimator(run_config, params, FLAGS.policies_dir)

    for policy_iteration in range(FLAGS.num_iterations):
        env = Soccer(k_last_models=3)
        print("iteration={}".format(policy_iteration))

        model_path = tf.train.latest_checkpoint(FLAGS.policies_dir)
        saver = tf.train.import_meta_graph(model_path + '.meta')

        with tf.Session() as sess:
            saver.restore(sess, model_path)
            graph = tf.get_default_graph()

            inputs = graph.get_tensor_by_name("shuffle_batch:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            output = graph.get_tensor_by_name("PolicyNetwork/output/BiasAdd:0")

            states = []
            actions = []
            rewards = []
            games_won = 0

            for _ in range(FLAGS.games_against_same_opponent):
                for game in range(FLAGS.num_games):

                    starting_game = np.random.rand() < 0.5
                    state = env.reset(starting_game=starting_game, verbose=0)

                    for turn in range(FLAGS.max_turns_per_game):
                        legal_moves = env.get_legal_moves()

                        if np.random.random() < FLAGS.exploratory_epsilon * (1 / (policy_iteration + 1) ** (1 / 2)):
                            action = int(np.random.random() * 8)
                        else:
                            feed_dict = {
                                inputs: state,
                                keep_prob: 1.0
                            }

                            logits = sess.run(output, feed_dict=feed_dict)
                            acts = sorted(range(len(logits[0])), key=lambda k: logits[0][k], reverse=True)

                            lgl_mvs = legal_moves
                            action = acts[0]
                            for act in acts:
                                if legal_moves[act] == 1:
                                    action = act
                                    break


                            # if action is None:
                            #     env.player_board.print_board()
                            #     print(legal_moves)
                            #     print(acts)

                            # illegal_moves = [100 * (1 - k) for k in legal_moves]
                            # illegal_moves_penalty = np.array(illegal_moves)
                            # logits = logits - illegal_moves_penalty
                            # print(env.get_legal_moves())

                        # env.player_board.print_board()
                        # print(legal_moves)
                        # print('player')
                        # print(action)
                        states.append(state[0])
                        # actions.append([1 if i == action else 0 for i in range(8)])
                        actions.append(action)
                        state, reward, done = env.step(action, verbose=0)

                        if done:
                            # print('Game ended, reward = {}'.format(reward))
                            rewards += [reward] * (turn + 1)
                            if reward == 1:
                                games_won += 1
                            break
                    # if reward == 1:
                        # env.player_board.print_board()
                        # print("Game won!")
                    if game % FLAGS.print_board == FLAGS.print_board - 1:
                        env.player_board.print_board()
                        # print(lgl_mvs)
                        # print(action)
                        # print(env.player_board.ball_pos)
                        # for i in range(8):
                        #     print(i)
                        #     env.player_board.print_layer(i)
                        print("Win ratio = {:.2f}%".format(100 * games_won / game))
                        # print(game)

                # state_actions_reward_history_arr = np.array(state_actions, dtype=np.int8)
                # np.random.shuffle(state_actions_reward_history_arr)
                num_state_actions = len(rewards)
                num_epochs = int(num_state_actions / FLAGS.batch_size)  # todo check if correct
                idx = np.random.choice(len(states), FLAGS.sample_size, replace=False)
                states2 = np.take(states, idx, axis=0)
                actions2 = np.take(actions, idx)
                rewards2 = np.take(rewards, idx)
                input_fn = lambda: get_input_fn(states, actions, rewards)
                policy_network.train(input_fn, steps=1)
