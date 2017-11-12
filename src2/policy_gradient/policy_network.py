import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys


def res_block(x, scope):
    kernels = 128

    net = slim.conv2d(x, kernels, [3, 3], padding='SAME', scope='{}_conv1'.format(scope))
    net = slim.batch_norm(net, activation_fn=tf.nn.relu)
    net = slim.conv2d(net, kernels, [3, 3], padding='SAME', scope='{}_conv2'.format(scope))
    net = tf.nn.relu(x + net)

    return net


class CnnPolicy:
    def __init__(self, sess, ob_shape, n_acts):
        # todo for now its [1,11,9,12], in future we might want to predict for more than 1 obs
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope('PolicyNetwork'):
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=slim.l2_regularizer(0.001)
            ):
                conv1 = slim.conv2d(X, 128, [3, 3], padding='SAME', scope='conv1')
                bn1 = slim.batch_norm(conv1, activation_fn=tf.nn.relu)

                res1 = res_block(bn1, scope='block1')
                res2 = res_block(res1, scope='block2')
                res3 = res_block(res2, scope='block3')

                flat = slim.flatten(res3)
                fc = slim.fully_connected(flat, 256, scope='fc1')
                pi = slim.fully_connected(fc, n_acts, scope='pi', activation_fn=None)
                vf = slim.fully_connected(fc, 1, scope='vf', activation_fn=None)

        v0 = vf[:, 0]
        a0 = tf.argmax(pi, axis=1)

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X: ob})
            return a, v, []  # dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
