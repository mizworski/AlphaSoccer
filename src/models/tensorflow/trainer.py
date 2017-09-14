"""Script to illustrate usage of tf.estimator.Estimator in TF v1.3"""
import tensorflow as tf
import os

# from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner

from src.models.dataset import Dataset

# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG)
GAMES_DIR = os.path.join('data', 'games')

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='model_dir', default_value='./training',
    docstring='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    flag_name='data_dir', default_value='./data',
    docstring='Directory to download the data to.')


# Define and run experiment ###############################
def run_experiment(argv=None):
    """Run the training experiment."""
    # Define model parameters
    params = tf.contrib.training.HParams(
        learning_rate=0.01,
        # n_classes=8,
        # train_steps=4,
        min_eval_frequency=25
    )

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)

    learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=params
    )


def experiment_fn(run_config, params):
    batch_size = 128
    run_config = run_config.replace(
        save_checkpoints_steps=params.min_eval_frequency)
    estimator = get_estimator(run_config, params)
    dataset = Dataset(GAMES_DIR, batch_size=batch_size)
    train_input_fn, train_input_hook = get_train_inputs(
        batch_size=batch_size, dataset=dataset)
    eval_input_fn, eval_input_hook = get_test_inputs(
        batch_size=batch_size, dataset=dataset)
    print("Train samples = {}".format(dataset.training_samples))
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=None,
        eval_steps=int(dataset.validation_samples / batch_size),
        min_eval_frequency=params.min_eval_frequency
    )
    return experiment


def get_estimator(run_config, params):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config
    )


def model_fn(features, labels, mode, params):
    is_training = mode == ModeKeys.TRAIN
    logits = architecture(features, is_training=is_training)
    predictions = tf.argmax(logits, axis=-1)
    labels = tf.argmax(labels, axis=-1)
    loss = None
    train_op = None
    eval_metric_ops = {}
    if mode != ModeKeys.INFER:
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(labels, tf.int32),
            logits=logits)
        train_op = get_train_op_fn(loss, params)
        eval_metric_ops = get_eval_metric_ops(labels, predictions)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def get_train_op_fn(loss, params):
    def _train_op():
        tf.add_check_numerics_ops()

        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=params.learning_rate
        )

    return _train_op()


def get_eval_metric_ops(labels, predictions):
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            name='accuracy')
    }


def architecture(inputs, is_training, scope='MnistConvNet'):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer()):
            net = slim.conv2d(inputs, 128, [5, 5], padding='SAME',
                              scope='conv1')
            net = slim.conv2d(net, 128, [3, 3], padding='SAME',
                              scope='conv2')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 256, scope='fn3')
            net = slim.dropout(net, is_training=is_training,
                               scope='dropout3')
            net = slim.fully_connected(net, 8, scope='output',
                                       activation_fn=None)
        return net


# Define data loaders #####################################
class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)


def get_train_inputs(batch_size, dataset):
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        with tf.name_scope('Training_data'):
            next_example, next_label = next(dataset.train_generator)
            next_example = tf.convert_to_tensor(next_example, tf.float32)
            next_label = tf.convert_to_tensor(next_label, tf.float32)
            return next_example, next_label

    return train_inputs, iterator_initializer_hook


def get_test_inputs(batch_size, dataset):
    iterator_initializer_hook = IteratorInitializerHook()

    def test_inputs():
        with tf.name_scope('Test_data'):
            next_example, next_label = next(dataset.validation_generator)
            next_example = tf.convert_to_tensor(next_example, tf.float32)
            next_label = tf.convert_to_tensor(next_label, tf.float32)
            return next_example, next_label

    return test_inputs, iterator_initializer_hook


if __name__ == "__main__":
    tf.app.run(
        main=run_experiment
    )
