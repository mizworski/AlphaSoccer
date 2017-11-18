import tensorflow as tf
from tensorflow.contrib import slim

n_kernels = 128


def res_block(x, scope, reuse):
    with tf.variable_scope(scope, reuse=reuse):
        net = slim.conv2d(x, n_kernels, [3, 3], padding='SAME', scope='conv1')
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)

        net = slim.conv2d(net, n_kernels, [3, 3], padding='SAME', scope='conv2')
        net = tf.nn.relu(x + net)

    return net


class CnnPolicy:
    def __init__(self, sess, ob_space, n_act, n_batch, scope, reuse=False):
        ob_shape = (n_batch,) + ob_space.shape
        X = tf.placeholder(tf.float32, ob_shape, name='state')
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=slim.l2_regularizer(0.001)
            ):
                conv1 = slim.conv2d(X, 128, [3, 3], padding='SAME', scope='conv1')
                bn1 = slim.batch_norm(conv1, activation_fn=tf.nn.relu)

                res1 = res_block(bn1, scope='res_block1', reuse=reuse)
                res2 = res_block(res1, scope='res_block2', reuse=reuse)
                res3 = res_block(res2, scope='res_block3', reuse=reuse)

                flat = slim.flatten(res3)
                fc = slim.fully_connected(flat, 256, scope='fc1')
                with tf.variable_scope('policy_head', reuse=reuse):
                    logits = slim.fully_connected(fc, n_act, scope='logits', activation_fn=None)
                    probs = tf.nn.softmax(logits, name='probs')
                with tf.variable_scope('value_head', reuse=reuse):
                    vf = slim.fully_connected(fc, 1, scope='vf', activation_fn=tf.nn.tanh)

        v0 = vf[:, 0]
        pi0 = probs[:]

        def step(ob, *_args, **_kwargs):
            pi, v = sess.run([pi0, v0], {X: ob})
            return pi, v

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.logits = logits
        self.pi = probs
        self.vf = vf
        self.step = step
        self.value = value