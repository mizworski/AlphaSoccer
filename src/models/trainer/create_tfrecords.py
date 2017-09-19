import tensorflow as tf
import pandas as pd
import os
from numpy import genfromtxt
import numpy as np
from functools import reduce
import operator

input_dim = [11, 9, 12]

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

subsets = ['train', 'eval']
for subset in subsets:
    games_dir = 'data/games/{}/'.format(subset)
    files = os.listdir(games_dir)
    files = [os.path.join(games_dir, file) for file in files]

    count = 0
    n_features = reduce(operator.mul, input_dim, 1)
    state_action_batch = np.ndarray((0, n_features + 1), dtype=np.int8)
    for file in files:
        csv = genfromtxt(file, delimiter=',', dtype=np.int8)
        state_action_batch = np.concatenate((state_action_batch, csv))
        count += 1
        if count % 100 == 0:
            print(count)
            # break

    np.random.shuffle(state_action_batch)
    # filename = "data/tfrecords/{}".format(subset)

    with tf.python_io.TFRecordWriter(subset) as writer:
        for row in state_action_batch:
            features, label = row[:-1], row[-1]
            features_string = features.tostring()
            features_compat_string = tf.compat.as_bytes(features_string)

            example = tf.train.Example(features=tf.train.Features(feature={
                'features': _bytes_feature(features_compat_string),
                'label': _int64_feature(label),
            }))
            writer.write(example.SerializeToString())
