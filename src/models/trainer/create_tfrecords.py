import tensorflow as tf
import pandas as pd
import os
from numpy import genfromtxt
import numpy as np

games_dir = 'data/games/train/'
files = os.listdir(games_dir)
files = [os.path.join(games_dir, file) for file in files]

count = 0
state_action_batch = np.ndarray((0, 1090), dtype=np.int8)
for file in files:
    csv = genfromtxt(file, delimiter=',', dtype=np.int8)
    state_action_batch = np.concatenate((state_action_batch, csv))
    count += 1
    if count % 100 == 0:
        print(count)

np.random.shuffle(state_action_batch)

with tf.python_io.TFRecordWriter("train") as writer:
    for row in state_action_batch:
        features, label = row[:-1], row[-1]
        example = tf.train.Example()
        example.features.feature["features"].int64_list.value.extend(features)
        example.features.feature["label"].int64_list.value.append(int(label))
        writer.write(example.SerializeToString())
