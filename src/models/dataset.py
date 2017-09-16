from numpy import genfromtxt
import numpy as np
import os
import keras

np.random.seed(42)

WIDTH, HEIGHT, DEPTH = 11, 9, 11
N_ACTIONS = 8

N_EPOCHS = 128
TRAINING_SET_SIZE = 0.7
GAMES_IN_BATCH = 256
N_STATE_ACTIONS = 282209


def split_dataset(games_dir):
    filenames = np.asarray(os.listdir(games_dir))
    np.random.shuffle(filenames)
    training_samples = int(filenames.shape[0] * TRAINING_SET_SIZE)

    return filenames[:training_samples], filenames[training_samples:]


class Dataset:
    def __init__(self, games_dir, batch_size=128, training_set_size=0.7):
        self.training_set_size = training_set_size
        self.batch_size = batch_size
        self.games_dir = games_dir
        self.train_files, self.validation_files = split_dataset(self.games_dir)

        self.train_generator = self.batch_generator(self.train_files)
        self.validation_generator = self.batch_generator(self.validation_files)

        self.training_samples = int(N_STATE_ACTIONS * self.training_set_size)
        self.validation_samples = int(N_STATE_ACTIONS * (1 - self.training_set_size))

    def batch_generator(self, filenames):
        n_games = len(filenames)

        while True:
            games_processed = 0
            state_action_batch = np.ndarray((0, 1090))
            for filename in filenames:
                file_path = os.path.join(self.games_dir, filename)
                state_actions = genfromtxt(file_path, delimiter=',')
                state_action_batch = np.concatenate((state_action_batch, state_actions))

                games_processed += 1
                if games_processed % GAMES_IN_BATCH == 0 or games_processed == n_games:
                    n_examples = state_action_batch.shape[0]
                    np.random.shuffle(state_action_batch)
                    states = state_action_batch[:, :-1]
                    actions = state_action_batch[:, -1:]

                    x = states.reshape(n_examples, WIDTH, HEIGHT, DEPTH)
                    y = keras.utils.to_categorical(actions, N_ACTIONS)

                    while x.shape[0] >= self.batch_size:
                        yield (x[:self.batch_size], y[:self.batch_size])
                        x = x[self.batch_size:]
                        y = y[self.batch_size:]

                    state_action_batch = np.ndarray((0, 1090))
