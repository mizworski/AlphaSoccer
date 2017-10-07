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
    flag_name='num_games', default_value='32',
    docstring='Number of games played between each policy iteration.'
)
tf.app.flags.DEFINE_integer(
    flag_name='batch_size', default_value='4',
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
tf.app.flags.DEFINE_float(
    flag_name='exploratory_epsilon', default_value='0.01',
    docstring='Epsilon for epsilon-greedy exploration.'
)

params = tf.contrib.training.HParams(
    learning_rate=0.00000001,
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
        logits = tf.Print(logits, [logits], message="logits:")
        log_prob = tf.log(tf.nn.softmax(logits))
        log_prob = tf.Print(log_prob, [log_prob], message="log_prob:")
        actions = features['actions']
        actions = tf.Print(actions, [actions], message="actions:")
        grad_act = tf.gather(log_prob, actions, axis=1)
        grad_act = tf.Print(grad_act, [grad_act], message="grad_act:")

        # actions = tf.cast(actions, dtype=tf.float32)
        # grad = tf.multiply(log_prob, actions)
        # grad = tf.Print(grad, [grad], message="grad:")
        # grad_act = tf.reduce_sum(grad, 1)

        real_grad = tf.multiply(grad_act, rewards)
        real_grad = tf.Print(real_grad, [real_grad], message="real_grad:")

        loss = -tf.reduce_sum(real_grad)

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
        env = Soccer(k_last_models=4)
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
            for game in range(FLAGS.num_games):

                starting_game = np.random.rand() < 0.5
                state = env.reset(starting_game=starting_game, verbose=0)

                for turn in range(FLAGS.max_turns_per_game):

                    if np.random.random() < FLAGS.exploratory_epsilon:
                        action = int(np.random.random() * 8)
                    else:
                        feed_dict = {
                            inputs: state,
                            keep_prob: 1.0
                        }

                        logits = sess.run([output], feed_dict=feed_dict)
                        action = np.argmax(logits)

                    states.append(state[0])
                    # actions.append([1 if i == action else 0 for i in range(8)])
                    actions.append(action)
                    state, reward, done = env.step(action, verbose=0)

                    if done:
                        # print('Game ended, reward = {}'.format(reward))
                        rewards += [reward] * (turn + 1)
                        break

                if game % 128 == 0:
                    print(game)
                    env.player_board.print_board()

            # state_actions_reward_history_arr = np.array(state_actions, dtype=np.int8)
            # np.random.shuffle(state_actions_reward_history_arr)
            num_state_actions = len(rewards)
            num_epochs = int(num_state_actions / FLAGS.batch_size)  # todo check if correct
            input_fn = lambda: get_input_fn(states, actions, rewards)
            policy_network.train(input_fn, steps=1)
