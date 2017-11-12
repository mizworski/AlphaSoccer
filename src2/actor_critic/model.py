import tensorflow as tf
from src2.actor_critic.policy_network import CnnPolicy
from src2.actor_critic.utils import cat_entropy, mse, Scheduler


class Model(object):
    def __init__(self, ob_space, ac_space, batch_size, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, lrschedule='linear', training_timesteps=int(80e6), verbose=0):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        n_act = ac_space.n

        A = tf.placeholder(tf.int32, [batch_size], name='action')
        ADV = tf.placeholder(tf.float32, [batch_size], name='advantage')
        R = tf.placeholder(tf.float32, [batch_size], name='reward')
        LR = tf.placeholder(tf.float32, [], name='learning_rate')

        step_model = CnnPolicy(sess, ob_space, n_act, 1, reuse=False)
        train_model = CnnPolicy(sess, ob_space, n_act, batch_size, reuse=True)

        logits = train_model.logits
        if verbose == 2:
            ADV_print = tf.Print(ADV, [ADV], summarize=8, message='advs: ')
            A_print = tf.Print(A, [A], summarize=8, message='actions: ')
            logits = tf.Print(logits, [logits], summarize=8, message='logits: ')
        else:
            A_print = A
            ADV_print = ADV

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=A_print)
        pg_loss = tf.reduce_mean(ADV_print * cross_entropy)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.logits))
        reg_loss = tf.losses.get_regularization_losses()

        if verbose == 2:
            pg_loss = tf.Print(pg_loss, [pg_loss], message='pg loss: ')
            vf_loss = tf.Print(vf_loss, [vf_loss], message='vf loss: ')
            reg_loss = tf.Print(reg_loss, [reg_loss], message='reg loss: ')

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + reg_loss

        if verbose == 2:
            loss = tf.Print(loss, [loss], message='loss: ')

        params = tf.trainable_variables(scope='AgentCriticNetwork')
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        # trainer = tf.train.GradientDescentOptimizer(learning_rate=LR)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=training_timesteps, schedule=lrschedule)

        def train(state, actions, rewards, values):
            advs = rewards - values
            cur_lr = lr.value()
            td_map = {train_model.X: state, A: actions, ADV: advs, R: rewards, LR: cur_lr}

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )

            return policy_loss, value_loss, policy_entropy

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        tf.global_variables_initializer().run(session=sess)
