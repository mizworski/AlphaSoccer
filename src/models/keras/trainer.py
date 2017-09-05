import os

import keras
import numpy as np
from numpy import genfromtxt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D

GAMES_DIR = os.path.join('data', 'games')
STARTING_INDEX = 1
N_EXAMPLES = 15432
BATCH_SIZE = 64

WIDTH, HEIGHT, DEPTH = 11, 9, 11
N_ACTIONS = 8

N_EPOCHS = 128
TRAINING_SET_SIZE = 0.7


def split_dataset(games_dir):
    filenames = np.asarray(os.listdir(games_dir))
    np.random.shuffle(filenames)
    training_samples = int(filenames.shape[0] * 0.7)

    return filenames[:training_samples], filenames[training_samples:]


def sracz_gen(filenames, batch_size):
    for batch_start_index in range(STARTING_INDEX, N_EXAMPLES + STARTING_INDEX, BATCH_SIZE):
        state_action_batch = np.ndarray((0, 1090))
        for i in range(batch_size):
            file_path = os.path.join(GAMES_DIR, "{}.csv".format(batch_start_index + i))
            state_actions = genfromtxt(file_path, delimiter=',')
            state_action_batch = np.concatenate((state_action_batch, state_actions))

        n_examples = state_action_batch.shape[0]
        np.random.shuffle(state_action_batch)
        states = state_action_batch[:, :-1]
        actions = state_action_batch[:, -1:]

        x = states.reshape(n_examples, WIDTH, HEIGHT, DEPTH)
        y = keras.utils.to_categorical(actions, N_ACTIONS)
        yield (x, y)


def batch_generator(filenames):
    n_games = len(filenames)

    games_processed = 0
    state_action_batch = np.ndarray((0, 1090))

    for filename in filenames:
        file_path = os.path.join(GAMES_DIR, filename)
        state_actions = genfromtxt(file_path, delimiter=',')
        state_action_batch = np.concatenate((state_action_batch, state_actions))

        games_processed += 1
        if games_processed % BATCH_SIZE == 0 or games_processed == n_games:
            n_examples = state_action_batch.shape[0]
            np.random.shuffle(state_action_batch)
            states = state_action_batch[:, :-1]
            actions = state_action_batch[:, -1:]

            x = states.reshape(n_examples, WIDTH, HEIGHT, DEPTH)
            y = keras.utils.to_categorical(actions, N_ACTIONS)
            yield (x, y)

            state_action_batch = np.ndarray((0, 1090))


def action_predictor():
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(WIDTH, HEIGHT, DEPTH),
                     padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))

    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(N_ACTIONS, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # generator = batch_generator(GAMES_DIR, BATCH_SIZE)
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    train_files, validation_files = split_dataset(GAMES_DIR)
    train_generator = batch_generator(train_files)
    validation_generator = batch_generator(validation_files)

    # with split_dataset(GAMES_DIR) as (train_files, validation_files):
    #     train_generator = batch_generator(train_files)
    #     validation_generator = batch_generator(validation_files)

    #
    # for x, y in generator:
    #     print(x.shape)
    #     print(y.shape)
    #
    model = action_predictor()

    steps_per_epoch = int(len(train_files) / BATCH_SIZE)
    validation_steps = int(len(validation_files) / BATCH_SIZE)
    model.fit_generator(train_generator, steps_per_epoch, N_EPOCHS,
                        validation_data=validation_generator, validation_steps=validation_steps)
    #
    # model.fit_generator()
