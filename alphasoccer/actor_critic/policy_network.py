import tensorflow as tf
from tensorflow.contrib import slim

last_layer_kernel_mult = 1


def res_block(net_input, scope, reuse, histograms, n_kernels):
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


def _resnet_block_v1(inputs, kernels, stride, name, projection_shortcut=None):
  with tf.variable_scope(name):
    shortcut = inputs
    kernels_last_layer = kernels

    if projection_shortcut is not None:
      # noinspection PyCallingNonCallable
      shortcut = projection_shortcut(inputs)
      shortcut = tf.layers.batch_normalization(shortcut)
      kernels_last_layer = last_layer_kernel_mult * kernels

    net = tf.contrib.layers.conv2d(inputs, num_outputs=kernels, kernel_size=3, stride=1, padding='SAME',
                                   activation_fn=None)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)

    net = tf.contrib.layers.conv2d(net, num_outputs=kernels_last_layer, kernel_size=3, stride=stride, padding='SAME',
                                   activation_fn=None)
    net = tf.layers.batch_normalization(net)

    net = net + shortcut
    net = tf.nn.relu(net)

  return net


def _resnet_block_v2(inputs, kernels, stride, name, projection_shortcut=None):
  with tf.variable_scope(name):
    if projection_shortcut is not None:
      # noinspection PyCallingNonCallable
      shortcut = projection_shortcut(inputs)
      shortcut = tf.layers.batch_normalization(shortcut)
    else:
      shortcut = tf.contrib.layers.conv2d(inputs, last_layer_kernel_mult * kernels, [1, 1], stride=1, padding='SAME',
                                          activation_fn=None,
                                          scope='projection_shortcut')

    net = tf.contrib.layers.conv2d(inputs, num_outputs=kernels, kernel_size=1, stride=1, padding='SAME',
                                   activation_fn=None)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)

    net = tf.contrib.layers.conv2d(net, num_outputs=kernels, kernel_size=3, stride=stride, padding='SAME',
                                   activation_fn=None)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)

    net = tf.contrib.layers.conv2d(net, num_outputs=last_layer_kernel_mult * kernels, kernel_size=1, stride=1,
                                   padding='SAME',
                                   activation_fn=None)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)

    net = net + shortcut

  return net


def _block_layer(net, kernels, num_blocks, name, building_block_type='resnet_v2'):
  def _projection_shortcut(inputs):
    return tf.contrib.layers.conv2d(inputs, last_layer_kernel_mult * kernels, [1, 1], stride=2, padding='SAME',
                                    activation_fn=None,
                                    scope='projection_shortcut')

  # building_block = _resnet_block_v2
  building_block = _resnet_block_v1

  with tf.variable_scope(name):
    for i in range(num_blocks - 1):
      net = building_block(net, kernels, stride=1, name='residual_block{}'.format(i))

    net = building_block(net, kernels, stride=2, name='residual_block{}'.format(num_blocks - 1),
                         projection_shortcut=_projection_shortcut)

  return net

# todo move block_sizes to params
class ResNetPolicy:
  def __init__(self, sess, ob_space, n_act, scope, reuse=False, histograms=False, n_kernels=128, reg_fact=1e-4,
               residual_blocks=8, block_sizes=(10, )):
    input_layer_shape = [None] + list(ob_space.shape)
    X = tf.placeholder(tf.float32, input_layer_shape, name='input_layer')

    with tf.variable_scope(scope, reuse=reuse):
      with tf.contrib.framework.arg_scope(
              [tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
              weights_initializer=tf.contrib.layers.xavier_initializer(),
              weights_regularizer=tf.contrib.layers.l2_regularizer(reg_fact)):
        with tf.variable_scope("initial_conv"):
          net_conv = tf.contrib.layers.conv2d(X, n_kernels, [3, 3], activation_fn=None, padding='SAME',
                                              scope='conv')
          net_bn = tf.layers.batch_normalization(net_conv)
          net = tf.nn.relu(net_bn)
          if histograms:
            tf.summary.histogram("{}/conv_block/conv".format(scope), net_conv)
            tf.summary.histogram("{}/conv_block/batch_norm".format(scope), net_bn)
            tf.summary.histogram("{}/conv_block/relu".format(scope), net)

        for i, num_blocks in enumerate(block_sizes):
          kernels = n_kernels * (2 ** i)
          net = _block_layer(net, kernels, num_blocks, name='resnet_block{}'.format(i))

        with tf.variable_scope('policy_head', reuse=reuse):
          policy_net = tf.contrib.layers.conv2d(net, num_outputs=2, kernel_size=1, activation_fn=None, padding='SAME',
                                                scope='conv')
          policy_net = tf.layers.batch_normalization(policy_net)
          policy_net = tf.nn.relu(policy_net)
          if histograms:
            tf.summary.histogram("{}/policy/relu".format(scope), policy_net)
          policy_net = tf.layers.flatten(policy_net)

          logits = tf.contrib.layers.fully_connected(policy_net, n_act, scope='logits', activation_fn=None)
          probs = tf.nn.softmax(logits, name='probs')

        with tf.variable_scope('value_head', reuse=reuse):
          value_net = tf.contrib.layers.conv2d(net, num_outputs=1, kernel_size=1, activation_fn=None, padding='SAME',
                                               scope='conv')
          value_net = tf.layers.batch_normalization(value_net)
          value_net = tf.nn.relu(value_net)
          if histograms:
            tf.summary.histogram("{}/value/relu".format(scope), value_net)
          value_net = tf.layers.flatten(value_net)

          value_net = tf.contrib.layers.fully_connected(value_net, num_outputs=256, activation_fn=None)
          value_net = tf.nn.relu(value_net)
          vf = tf.contrib.layers.fully_connected(value_net, num_outputs=1, activation_fn=tf.nn.tanh)

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


class CnnPolicy:
  def __init__(self, sess, ob_space, n_act, scope, reuse=False, histograms=False, n_kernels=128, reg_fact=1e-4,
               residual_blocks=8):
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

        for i in range(residual_blocks):
          block_scope = 'res_block_{}'.format(i)
          net = res_block(net, n_kernels=n_kernels, scope=block_scope, reuse=reuse, histograms=histograms)

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
