import tensorflow as tf
from tensorflow.contrib import slim

def     get_policy_network(inputs, is_training=False, reuse=None, scope='PolicyNetwork'):
    kernels = 128
    keep_prob = tf.placeholder_with_default(0.85, shape=(), name='keep_prob')

    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=slim.l2_regularizer(0.1)
        ):
            net = slim.conv2d(inputs, kernels, [5, 5], padding='SAME',
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
            net = slim.flatten(net)
            net = slim.fully_connected(net, 256,
                                       reuse=reuse,
                                       scope='fc6')
            net = slim.dropout(net, is_training=is_training,
                               keep_prob=keep_prob,
                               scope='dropout6')
            net = slim.fully_connected(net, 8,
                                       scope='output',
                                       reuse=reuse,
                                       activation_fn=None)
        return net