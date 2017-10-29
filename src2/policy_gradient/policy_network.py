import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys


def res_block(x, reuse, scope):
    kernels = 128

    net = slim.conv2d(x, kernels, [3, 3], padding='SAME', reuse=reuse, scope='{}_conv1'.format(scope))
    net = slim.batch_norm(net, activation_fn=tf.nn.relu)
    net = slim.conv2d(net, kernels, [3, 3], padding='SAME', reuse=reuse, scope='{}_conv2'.format(scope))
    net = tf.nn.relu(tf.add(net, x))

    return net

def get_policy_network(inputs, reuse=None, scope='SLNet'):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=slim.l2_regularizer(0.1)
        ):
            net = slim.conv2d(inputs, 128, [3, 3], padding='SAME', reuse=reuse, scope='conv1')
            net = slim.batch_norm(net, activation_fn=tf.nn.relu)

            net = res_block(net, reuse, scope='block1')
            net = res_block(net, reuse, scope='block2')
            net = res_block(net, reuse, scope='block3')
            net = res_block(net, reuse, scope='block4')

            net = slim.flatten(net)
            net = slim.fully_connected(net, 256,
                                       reuse=reuse,
                                       scope='fc6')
            net = slim.fully_connected(net, 8,
                                       scope='output',
                                       reuse=reuse,
                                       activation_fn=None)
        return net


def get_estimator(run_config, params, model_dir):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config,
        model_dir=model_dir
    )


def model_fn(features, labels, mode, params):
    # rewards = labels
    logits = get_policy_network(features['states'])
    network_outputs = {
        'probabilities': tf.nn.softmax(logits),
        'labels': tf.argmax(logits, axis=-1)
    }

    loss = None
    train_op = None
    if mode != ModeKeys.INFER:
        loss = get_loss_reinforcement_learning(network_outputs['probabilities'], labels)
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
        optimizer=tf.train.AdamOptimizer,
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
    loss = -tf.reduce_sum(gain_single_action)
    return loss