"""Script to illustrate usage of tf.estimator.Estimator in TF v1.3"""
import tensorflow as tf
import os
import numpy as np

from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner
import multiprocessing
from tensorflow.python.lib.io import file_io

# from src.models.dataset import Dataset

tf.logging.set_verbosity(tf.logging.DEBUG)
# GAMES_DIR = os.path.join('data', 'games')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    flag_name='model_dir', default_value='model',
    docstring='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    flag_name='train_data_dir', default_value='data/games/train',
    docstring='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    flag_name='eval_data_dir', default_value='data/games/eval',
    docstring='Output directory for model and training stats.')

tf.app.flags.DEFINE_integer(
    flag_name='num_epochs', default_value='1024',
    docstring='epochs number'
)
tf.app.flags.DEFINE_integer(
    flag_name='batch_size', default_value='256',
    docstring='batch size'
)

tf.app.flags.DEFINE_string(
    flag_name='train_files', default_value=None,
    docstring='train files'
)
tf.app.flags.DEFINE_string(
    flag_name='eval_files', default_value=None,
    docstring='eval files'
)

params = tf.contrib.training.HParams(
    learning_rate=0.001,
    min_eval_frequency=1,
    save_checkpoints_steps=1,
    num_epochs=FLAGS.num_epochs,
    batch_size=FLAGS.batch_size
)
def run_experiment(argv=None):

    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)

    learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=params
    )


def experiment_fn(run_config, params):
    if FLAGS.train_files is None:
        train_files_names = os.listdir(FLAGS.train_data_dir)
        train_files = [os.path.join(FLAGS.train_data_dir, filename) for filename in train_files_names]
    else:
        # with file_io.FileIO(FLAGS.train_files, mode='r') as file:
        #     train_files = file.readlines()
        # train_files = [file_io.FileIO(FLAGS.train_files, mode='r')]
        train_files = [FLAGS.train_files]

    train_input_fn = lambda: generate_input_fn(
        train_files,
        num_epochs=params.num_epochs,
        batch_size=params.batch_size
    )
    if FLAGS.train_files is None:
        eval_files = np.asarray(os.listdir(FLAGS.eval_data_dir))
        eval_files = [os.path.join(FLAGS.eval_data_dir, filename) for filename in eval_files]
    else:
        # with file_io.FileIO(FLAGS.eval_files, mode='r') as file:
        #     eval_files = file.readlines()
        # eval_files = [file_io.FileIO(FLAGS.train_files, mode='r')]
        eval_files = [FLAGS.eval_files]

    eval_input_fn = lambda: generate_input_fn(
        eval_files,
        num_epochs=params.num_epochs,
        batch_size=params.batch_size,
        shuffle=False
    )

    avg_actions_per_game = 112

    training_steps = int(len(train_files) * avg_actions_per_game / params.batch_size)
    validation_steps = int(len(eval_files) * avg_actions_per_game / params.batch_size)

    run_config = run_config.replace(
        save_checkpoints_steps=params.save_checkpoints_steps * training_steps
    )
    estimator = get_estimator(run_config, params)

    print("Train samples = {}".format(len(train_files) * avg_actions_per_game))

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=None,
        eval_steps=validation_steps,
        min_eval_frequency=params.min_eval_frequency * training_steps
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

    labels = tf.cast(labels, tf.int32)
    loss = None
    train_op = None
    eval_metric_ops = {}
    if mode != ModeKeys.INFER:
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
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
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.learning_rate
    )


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
                weights_initializer=tf.contrib.layers.xavier_initializer()
        ):
            net = slim.conv2d(inputs, 128, [5, 5], padding='SAME',
                              scope='conv1')
            net = slim.conv2d(net, 128, [3, 3], padding='SAME',
                              scope='conv2')
            net = slim.conv2d(net, 128, [3, 3], padding='SAME',
                              scope='conv3')
            net = slim.conv2d(net, 128, [3, 3], padding='SAME',
                              scope='conv4')
            # net = slim.conv2d(net, 128, [3, 3], padding='SAME',
            #                   scope='conv5')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 256, scope='fn6')
            # net = slim.dropout(net, is_training=is_training,
            #                    scope='dropout6')
            net = slim.fully_connected(net, 8, scope='output',
                                       activation_fn=None)
        return net


def parse_file(rows):
    row_cols = tf.expand_dims(rows, -1)
    columns = tf.decode_csv(row_cols, record_defaults=[[0.0]] * 1090)
    features = tf.reshape(tf.stack(columns[:-1]), [-1, 11, 9, 11])
    labels = tf.reshape(columns[-1:], [-1, 1])
    return [features, labels]


def generate_input_fn(filenames,
                      num_epochs=None,
                      shuffle=True,
                      batch_size=256):
    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=shuffle
    )
    reader = tf.TextLineReader()

    _, rows = reader.read_up_to(filename_queue, num_records=batch_size)
    batch = parse_file(rows)

    if shuffle:
        batch = tf.train.shuffle_batch(
            batch,
            batch_size,
            min_after_dequeue=2 * batch_size + 1,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=True,
            # shapes=[1, 11, 9, 11]
        )
    else:
        batch = tf.train.batch(
            batch,
            batch_size,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=True
        )

    return batch[0], batch[1]


if __name__ == "__main__":
    tf.app.run(
        main=run_experiment
    )
