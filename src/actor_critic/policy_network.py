import os
import tensorflow as tf
from tensorflow.contrib import slim

n_kernels = 128
reg_fact = 1e-4
n_residual_blocks = 8
log_dir = 'models/logs/'


def res_block(net_input, scope, reuse):
    with tf.variable_scope(scope, reuse=reuse):
        net = slim.conv2d(net_input, n_kernels, [3, 3], padding='SAME', scope='conv_1')
        net = slim.batch_norm(net)
        net = tf.nn.relu(net)

        net = slim.conv2d(net, n_kernels, [3, 3], padding='SAME', scope='conv_2')
        net = slim.batch_norm(net)
        net = tf.add(net, net_input)
        net = tf.nn.relu(net)

    return net


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class CnnPolicy:
    def __init__(self, sess, ob_space, n_act, scope, reuse=False):
        ob_shape = [None] + list(ob_space.shape)
        X = tf.placeholder(tf.float32, ob_shape, name='state')
        step_writer = tf.summary.FileWriter(os.path.join(log_dir, scope), sess.graph)

        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=slim.l2_regularizer(reg_fact)
            ):
                with tf.variable_scope('conv_block', reuse=reuse):
                    net = slim.conv2d(X, n_kernels, [3, 3], padding='SAME')
                    net = slim.batch_norm(net)
                    net = tf.nn.relu(net)

                for i in range(n_residual_blocks):
                    net = res_block(net, scope='res_block_{}'.format(i), reuse=reuse)

                with tf.variable_scope('policy_head', reuse=reuse):
                    policy_net = slim.conv2d(net, num_outputs=2, kernel_size=[1, 1])
                    policy_net = slim.batch_norm(policy_net)
                    policy_net = tf.nn.relu(policy_net)
                    policy_net = slim.flatten(policy_net)

                    logits = slim.fully_connected(policy_net, n_act, scope='logits', activation_fn=None)
                    probs = tf.nn.softmax(logits, name='probs')
                with tf.variable_scope('value_head', reuse=reuse):
                    value_net = slim.conv2d(net, num_outputs=1, kernel_size=[1, 1])
                    value_net = slim.batch_norm(value_net)
                    value_net = tf.nn.relu(value_net)
                    value_net = slim.flatten(value_net)

                    vf = slim.fully_connected(value_net, num_outputs=1, activation_fn=tf.nn.tanh)

        v0 = vf
        pi0 = probs
        self.i = 0

        def step(ob, *_args, **_kwargs):
            pi, v = sess.run([pi0, v0], {X: ob})
            return pi, v

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        # self.batch_size = batch_size
        self.X = X
        self.logits = logits
        self.pi = probs
        self.vf = vf
        self.step = step
        self.value = value
