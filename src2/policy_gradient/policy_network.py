import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys


def get_policy_network(inputs, reuse=None, scope='PolicyNetwork'):
    kernels = 128
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=slim.l2_regularizer(0.2)
        ):
            net = slim.conv2d(inputs, kernels, [3, 3], padding='SAME',
                              reuse=reuse,
                              scope='conv1')
            net = slim.conv2d(net, kernels, [3, 3], padding='SAME',
                              reuse=reuse,
                              scope='conv2')
            net = slim.conv2d(net, kernels, [3, 3], padding='SAME',
                              reuse=reuse,
                              scope='conv3')
            net = slim.conv2d(net, kernels, [3, 3], padding='SAME',
                              reuse=reuse,
                              scope='conv4')
            net = slim.conv2d(net, kernels, [3, 3], padding='SAME',
                              reuse=reuse,
                              scope='conv5')
            net = slim.conv2d(net, kernels, [3, 3], padding='SAME',
                              reuse=reuse,
                              scope='conv6')
            net = slim.conv2d(net, kernels, [3, 3], padding='SAME',
                              reuse=reuse,
                              scope='conv7')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 256,
                                       reuse=reuse,
                                       scope='fc8')
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
    eval_metric_ops = None
    if mode != ModeKeys.INFER:
        loss = get_loss_supervised_learning(network_outputs['probabilities'], labels)
        train_op = get_train_op_fn(loss, params)
        eval_metric_ops = get_eval_metric_ops(labels, logits, network_outputs['labels'])

    predictions = network_outputs if mode == ModeKeys.INFER else network_outputs['labels']
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
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

def get_loss_supervised_learning(logits, labels):
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits)
    return loss

def get_eval_metric_ops(labels, logits, predictions):
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            name='accuracy'),
        'Top2': tf.metrics.recall_at_k(
            labels=labels,
            predictions=logits,
            k=2,
            name='in_top2'
        ),
        'Top3': tf.metrics.recall_at_k(
            labels=labels,
            predictions=logits,
            k=3,
            name='in_top3'
        ),
        'Top4': tf.metrics.recall_at_k(
            labels=labels,
            predictions=logits,
            k=4,
            name='in_top4'
        ),
    }
