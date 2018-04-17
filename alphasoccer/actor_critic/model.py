import os
import sys
import re

import tensorflow as tf

from alphasoccer.actor_critic.policy_network import CnnPolicy, ResNetPolicy
from alphasoccer.actor_critic.utils import Scheduler


class Model(object):
  def __init__(self, ob_space, ac_space, batch_size, vf_coef=1.0, max_grad_norm=2.0, lr=1e-3,
               lrschedule='constant', training_timesteps=1024, model_dir='models/actor_critic',
               log_dir='models/logs/test/', momentum=0.9, n_kernels=128, reg_fact=1e-4, residual_blocks=8):

    training_player_scope = 'training_player'
    best_player_scope = 'best_player'

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    n_act = ac_space.n

    PI = tf.placeholder(tf.float32, [batch_size, n_act], name='pi')
    R = tf.placeholder(tf.float32, [batch_size], name='reward')
    LR = tf.placeholder(tf.float32, [], name='learning_rate')

    basic_summaries_list = []

    step_model = ResNetPolicy(sess, ob_space, n_act, best_player_scope, reuse=False, n_kernels=n_kernels,
                           reg_fact=reg_fact, residual_blocks=residual_blocks)
    train_model = ResNetPolicy(sess, ob_space, n_act, training_player_scope, reuse=False, histograms=True,
                            n_kernels=n_kernels, reg_fact=reg_fact, residual_blocks=residual_blocks)
    with tf.variable_scope('loss'):
      with tf.variable_scope('actor_loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_model.logits, labels=PI)
        pg_loss = tf.reduce_mean(cross_entropy)
        basic_summaries_list.append(tf.summary.scalar('policy_cross_entropy', pg_loss))

      with tf.variable_scope('critic_loss'):
        predictions = tf.squeeze(train_model.vf, axis=1)
        vf_loss = tf.losses.mean_squared_error(predictions=predictions, labels=R)
        basic_summaries_list.append(tf.summary.scalar('critic_mse', vf_loss))

      with tf.variable_scope('regularization_loss'):
        reg_loss = tf.reduce_mean(tf.losses.get_regularization_losses(scope='training_player'))
        basic_summaries_list.append(tf.summary.scalar('reg_loss', reg_loss))

      # loss = pg_loss + vf_loss * vf_coef + reg_loss
      loss = vf_loss
      basic_summaries_list.append(tf.summary.scalar('total_loss', loss))

    params = tf.trainable_variables(scope=training_player_scope)
    grads = tf.gradients(loss, params)
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    grads = list(zip(grads, params))

    basic_summaries = tf.summary.merge(basic_summaries_list)
    detailed_summaries = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # trainer = tf.train.MomentumOptimizer(learning_rate=LR, momentum=momentum)
    trainer = tf.train.AdamOptimizer(learning_rate=LR)
    _train = trainer.apply_gradients(grads)

    self.lr = Scheduler(v=lr, n_values=training_timesteps, schedule=lrschedule)

    saver = tf.train.Saver()

    def train(state, pi, rewards, train_iter=0):
      cur_lr = self.lr.value()
      td_map = {
        train_model.X: state,
        PI: pi,
        R: rewards,
        LR: cur_lr
      }

      if train_iter % 256 == 0:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, policy_loss, value_loss, _ = sess.run([detailed_summaries, pg_loss, vf_loss, _train],
                                                       options=run_options,
                                                       run_metadata=run_metadata,
                                                       feed_dict=td_map)
        self.summary_writer.add_run_metadata(run_metadata, 'step%03d' % train_iter)
        self.summary_writer.add_summary(summary, train_iter)
      else:
        summary, policy_loss, value_loss, _ = sess.run(
          [basic_summaries, pg_loss, vf_loss, _train],
          td_map
        )
        self.summary_writer.add_summary(summary, train_iter)

      return policy_loss, value_loss

    def save_model(step=0):
      model_path = os.path.join(model_dir, 'model.ckpt')
      saver.save(sess, model_path, global_step=step)
      print('Successfully saved step={}'.format(step))

    def update_best_player():
      best_player_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=best_player_scope)
      train_player_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=training_player_scope)

      copy_vars = []

      for best_model_var, train_player_var in zip(best_player_vars, train_player_vars):
        copy_var = best_model_var.assign(train_player_var)
        copy_vars.append(copy_var)

      sess.run(copy_vars)

    self.train = train
    self.train_model = train_model
    self.step_model = step_model
    self.step = step_model.step
    self.value = step_model.value

    self.save = save_model
    self.update_best_player = update_best_player
    self.sess = sess

    latest_checkpoint = tf.train.latest_checkpoint(model_dir)

    if latest_checkpoint is None:
      tf.global_variables_initializer().run(session=sess)
      self.initial_checkpoint_number = 1
      print('No checkpoint found. Starting new model.', file=sys.stderr)
    else:
      saver.restore(sess, save_path=latest_checkpoint)
      self.initial_checkpoint_number = int(re.findall(r'\d+', latest_checkpoint)[-1])
      print('Loaded checkpoint {}'.format(latest_checkpoint), file=sys.stderr)

    # initialize training player with current best player
    self.update_best_player()
