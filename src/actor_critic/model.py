import os
import tensorflow as tf
from src.actor_critic.policy_network import CnnPolicy
from src.actor_critic.utils import cat_entropy, mse, Scheduler


class Model(object):
    def __init__(self, ob_space, ac_space, batch_size, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=1e-8,
                 alpha=0.99, epsilon=1e-5, lrschedule='linear', training_timesteps=int(1e6),
                 model_dir='models/actor_critic', verbose=0):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        n_act = ac_space.n

        A = tf.placeholder(tf.int32, [batch_size], name='action')
        ADV = tf.placeholder(tf.float32, [batch_size], name='advantage')
        R = tf.placeholder(tf.float32, [batch_size], name='reward')
        LR = tf.placeholder(tf.float32, [], name='learning_rate')

        training_player_scope = 'training_player'
        best_player_scope = 'best_player'

        step_model = CnnPolicy(sess, ob_space, n_act, 1, best_player_scope, reuse=False)
        train_model = CnnPolicy(sess, ob_space, n_act, batch_size, training_player_scope, reuse=False)


        with tf.variable_scope('loss'):
            logits = train_model.logits

            with tf.variable_scope('actor_loss'):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=A)
                pg_loss = tf.reduce_mean(ADV * cross_entropy)

            with tf.variable_scope('critic_loss'):
                vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))

            with tf.variable_scope('regularization_loss'):
                entropy = tf.reduce_mean(cat_entropy(train_model.logits))
                reg_loss = tf.losses.get_regularization_losses()

            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + reg_loss

        params = tf.trainable_variables(scope=training_player_scope)
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        # trainer = tf.train.GradientDescentOptimizer(learning_rate=LR)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=training_timesteps, schedule=lrschedule)

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(model_dir, sess.graph)

        self.training_timestep = 0
        def train(state, actions, rewards):
            advs = rewards
            cur_lr = lr.value()
            td_map = {train_model.X: state, A: actions, ADV: advs, R: rewards, LR: cur_lr}

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )

            return policy_loss, value_loss, policy_entropy

        def save_model(step=0):
            model_path = os.path.join(model_dir, 'model.ckpt')
            saver.save(sess, model_path, global_step=step)

        def update_best_player():
            best_player_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=best_player_scope)
            train_player_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=training_player_scope)

            # assuming all variables are in the same order
            # todo check
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

        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint is None:
            tf.global_variables_initializer().run(session=sess)
        else:
            saver.restore(sess, save_path=latest_checkpoint)