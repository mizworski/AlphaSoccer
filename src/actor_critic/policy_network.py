import tensorflow as tf
from tensorflow.contrib import slim

n_kernels = 128
reg_fact = 1e-3
n_residual_blocks = 8
log_dir = 'models/logs/'


def res_block(net_input, scope, reuse, histograms):
    with tf.variable_scope(scope, reuse=reuse):
        net_conv1 = slim.conv2d(net_input, n_kernels, [3, 3], padding='SAME', scope='conv_1')
        net_bn1 = slim.batch_norm(net_conv1)
        net_relu1 = tf.nn.relu(net_bn1)

        net_conv2 = slim.conv2d(net_relu1, n_kernels, [3, 3], padding='SAME', scope='conv_2')
        net_bn2 = slim.batch_norm(net_conv2)
        net_added_input = tf.add(net_bn2, net_input)
        net_relu2 = tf.nn.relu(net_added_input)

        if histograms:
            tf.summary.histogram("{}/conv1".format(scope), net_conv1)
            tf.summary.histogram("{}/bn1".format(scope), net_bn1)
            tf.summary.histogram("{}/relu1".format(scope), net_relu1)

            tf.summary.histogram("{}/conv2".format(scope), net_conv2)
            tf.summary.histogram("{}/bn2".format(scope), net_bn2)
            tf.summary.histogram("{}/added_input".format(scope), net_added_input)
            tf.summary.histogram("{}/relu2".format(scope), net_relu2)

    return net_relu2


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
    def __init__(self, sess, ob_space, n_act, scope, reuse=False, histograms=False):
        input_layer_shape = [None] + list(ob_space.shape)
        X = tf.placeholder(tf.float32, input_layer_shape, name='input_layer')

        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=slim.l2_regularizer(reg_fact)
            ):
                with tf.variable_scope('conv_block', reuse=reuse):
                    net_conv = slim.conv2d(X, n_kernels, [3, 3], padding='SAME')
                    net_bn = slim.batch_norm(net_conv)
                    net = tf.nn.relu(net_bn)
                    if histograms:
                        tf.summary.histogram("{}/conv_block/conv".format(scope), net_conv)
                        tf.summary.histogram("{}/conv_block/batch_norm".format(scope), net_bn)
                        tf.summary.histogram("{}/conv_block/relu".format(scope), net)

                for i in range(n_residual_blocks):
                    block_scope = 'res_block_{}'.format(i)
                    net = res_block(net, scope=block_scope, reuse=reuse, histograms=histograms)

                with tf.variable_scope('policy_head', reuse=reuse):
                    policy_net = slim.conv2d(net, num_outputs=2, kernel_size=[1, 1])
                    policy_net = slim.batch_norm(policy_net)
                    policy_net = tf.nn.relu(policy_net)
                    if histograms:
                        tf.summary.histogram("{}/policy/relu".format(scope), policy_net)
                    policy_net = slim.flatten(policy_net)

                    logits = slim.fully_connected(policy_net, n_act, scope='logits', activation_fn=None)
                    probs = tf.nn.softmax(logits, name='probs')

                with tf.variable_scope('value_head', reuse=reuse):
                    value_net = slim.conv2d(net, num_outputs=1, kernel_size=[1, 1])
                    value_net = slim.batch_norm(value_net)
                    value_net = tf.nn.relu(value_net)
                    if histograms:
                        tf.summary.histogram("{}/value/relu".format(scope), value_net)
                    value_net = slim.flatten(value_net)

                    vf = slim.fully_connected(value_net, num_outputs=1, activation_fn=tf.nn.tanh)

        v0 = vf
        pi0 = probs
        self.i = 0

        def step(state):
            pi, v = sess.run([pi0, v0], {X: state})
            return pi, v

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.logits = logits
        self.pi = probs
        self.vf = vf
        self.step = step
        self.value = value
