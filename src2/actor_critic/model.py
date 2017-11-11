import tensorflow as tf
from src2.actor_critic.policy_network import CnnPolicy
from src2.actor_critic.utils import cat_entropy, mse, Scheduler


class Model(object):
    def __init__(self, ob_space, ac_space, n_batch, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, lrschedule='linear', total_timesteps=int(80e6)):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        n_act = ac_space.n

        A = tf.placeholder(tf.int32, [n_batch])
        ADV = tf.placeholder(tf.float32, [n_batch])
        R = tf.placeholder(tf.float32, [n_batch])
        LR = tf.placeholder(tf.float32, [])

        step_model = CnnPolicy(sess, ob_space, n_act, 1, reuse=False)
        train_model = CnnPolicy(sess, ob_space, n_act, n_batch, reuse=True)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * cross_entropy)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = tf.trainable_variables(scope='AgentCriticNetwork')

        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, rewards, actions, values):
            advs = rewards - values
            cur_lr = lr.value()
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: rewards, LR: cur_lr}

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