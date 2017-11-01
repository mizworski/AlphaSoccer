import tensorflow as tf
import os
import numpy as np

from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner
import multiprocessing
from functools import reduce
import operator

from src2.policy_gradient.policy_network import get_policy_network

tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    flag_name='model_dir', default_value='models/policy_networks/supervised_3',
    docstring='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    flag_name='train_data_dir', default_value='data/games/train',
    docstring='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    flag_name='eval_data_dir', default_value='data/games/eval',
    docstring='Output directory for model and training stats.')

tf.app.flags.DEFINE_integer(
    flag_name='num_epochs', default_value='128',
    docstring='epochs number'
)
tf.app.flags.DEFINE_integer(
    flag_name='batch_size', default_value='2048',
    docstring='batch size'
)

tf.app.flags.DEFINE_string(
    flag_name='train_files', default_value='data/tfrecords/train',
    docstring='train files'
)
tf.app.flags.DEFINE_string(
    flag_name='eval_files', default_value='data/tfrecords/eval',
    docstring='eval files'
)

params = tf.contrib.training.HParams(
    learning_rate=0.001,
    min_eval_frequency=1,
    save_checkpoints_steps=1,
    num_epochs=FLAGS.num_epochs,
    batch_size=FLAGS.batch_size
)

INPUT_SHAPE = [-1, 11, 9, 12]


def generate_input_fn(filenames,
                      num_epochs=None,
                      shuffle=True,
                      batch_size=256):
    feature = {'features': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=shuffle
    )
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)

    game = tf.to_float(tf.decode_raw(features['features'], tf.int8))
    label = tf.cast(features['label'], tf.int64)

    game = tf.reshape(game, INPUT_SHAPE)
    label = tf.reshape(label, [-1, 1])
    batch = [game, label]

    if shuffle:
        batch = tf.train.shuffle_batch(
            batch,
            batch_size,
            min_after_dequeue=2 * batch_size + 1,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=True,
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


def experiment_fn(run_config, params):
    train_files = [FLAGS.train_files]
    eval_files = [FLAGS.eval_files]

    train_input_fn = lambda: generate_input_fn(
        train_files,
        num_epochs=params.num_epochs,
        batch_size=params.batch_size
    )

    eval_input_fn = lambda: generate_input_fn(
        eval_files,
        num_epochs=params.num_epochs,
        batch_size=params.batch_size,
        shuffle=False
    )

    avg_actions_per_game = 112
    training_steps = int(2048 * avg_actions_per_game / params.batch_size)
    validation_steps = int((2495 - 2048) * avg_actions_per_game / params.batch_size)

    run_config = run_config.replace(
        save_checkpoints_steps=params.save_checkpoints_steps * training_steps - 1
    )
    estimator = get_estimator(run_config, params, FLAGS.model_dir)

    print("Train steps = {}".format(training_steps))
    print("Eval steps = {}".format(validation_steps))

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=training_steps * params.num_epochs,
        eval_steps=validation_steps,
        min_eval_frequency=params.min_eval_frequency * training_steps
    )
    return experiment


def get_estimator(run_config, params, model_dir):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config,
        model_dir=model_dir
    )


def model_fn(features, labels, mode, params):
    logits = get_policy_network(features)
    labels = tf.cast(labels, tf.int64)

    network_outputs = {
        'probabilities': tf.nn.softmax(logits),
        'labels': tf.argmax(logits, axis=-1)
    }

    loss = None
    train_op = None
    eval_metric_ops = None
    if mode != ModeKeys.INFER:
        loss = get_loss_supervised_learning(logits, labels)
        train_op = get_train_op_fn(loss, params)
        eval_metric_ops = get_eval_metric_ops(labels, logits, network_outputs['labels'])

    predictions = network_outputs['labels']
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
        # optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
        learning_rate=params.learning_rate
    )


def get_loss_supervised_learning(logits, labels):
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits)
    return loss


def get_eval_metric_ops(labels, logits, predictions):
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            name='accuracy'),
        'Top2': tf.metrics.recall_at_k(
            labels=labels,
            predictions=logits,
            k=2,
            name='in_top2'
        ),
        'Top3': tf.metrics.recall_at_k(
            labels=labels,
            predictions=logits,
            k=3,
            name='in_top3'
        ),
        'Top4': tf.metrics.recall_at_k(
            labels=labels,
            predictions=logits,
            k=4,
            name='in_top4'
        ),
    }

def run_experiment(argv=None):
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)

    learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=params
    )

if __name__ == "__main__":
    tf.app.run(
        main=run_experiment
    )
