import os
import pickle
from collections import namedtuple
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.environment.Board import print_board

Transition = namedtuple('Transition', ('state', 'pi', 'reward'))

schedules = {
    'constant': lambda p: 1,
    'linear': lambda p: 1 - p
}


class ReplayMemory(object):
    def __init__(self, capacity, replay_checkpoint_dir=os.path.join('data', 'replays'),
                 n_games_in_replay_checkpoint=100, verbose=0):
        self.capacity = capacity
        self.replay_checkpoint_dir = replay_checkpoint_dir
        self.memory, self.position = load_replays(replay_checkpoint_dir, capacity)
        self.verbose = verbose
        self.n_games_in_replay_checkpoint = n_games_in_replay_checkpoint

        if verbose:
            print("Loaded {} games from {}".format(len(self.memory), self.replay_checkpoint_dir))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)

        if self.verbose == 2:
            print("Reward : {}".format(self.memory[self.position].reward))
            print("Pi: {}".format(self.memory[self.position].pi))
            print("State")
            print_board(self.memory[self.position].state)
            print("*" * 8)

        self.position = (self.position + 1) % self.capacity

        if self.position % self.n_games_in_replay_checkpoint == 0:
            time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            output_file_name = 'checkpoint_{}_{}.pickle'.format(time_str, self.position)
            output_file_path = os.path.join(self.replay_checkpoint_dir, output_file_name)

            with open(output_file_path, 'wb') as file:
                if self.position == 0:
                    memory_slice = self.memory[-self.n_games_in_replay_checkpoint:]
                else:
                    memory_slice = self.memory[self.position - self.n_games_in_replay_checkpoint:self.position]

                pickle.dump(memory_slice, file)

    def push_vector(self, sars):
        transitions = [Transition(*sar) for sar in sars]

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transitions

        if self.verbose == 2:
            print("Transitions in history = {}".format(len(transitions)))

            i = 0
            for transition in self.memory[self.position]:
                print("Transition {}".format(i))
                print("Reward : {}".format(transition.reward))
                print("Pi: {}".format(transition.pi))
                print("State")
                print_board(transition.state)
                print("*" * 8)
                i += 1
                if i >= 8:
                    break

        self.position = (self.position + 1) % self.capacity

        if self.position % self.n_games_in_replay_checkpoint == 0:
            time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            output_file_name = 'checkpoint_{}_{}.pickle'.format(time_str, self.position)
            output_file_path = os.path.join(self.replay_checkpoint_dir, output_file_name)

            with open(output_file_path, 'wb') as file:
                if self.position == 0:
                    memory_slice = self.memory[-self.n_games_in_replay_checkpoint:]
                else:
                    memory_slice = self.memory[self.position - self.n_games_in_replay_checkpoint:self.position]

                pickle.dump(memory_slice, file)

    def sample(self, batch_size):
        replace = len(self.memory) < batch_size
        games = np.take(self.memory,
                        indices=np.random.choice(len(self.memory), size=batch_size, replace=replace),
                        axis=0)
        transitions = [np.take(game, np.random.choice(len(game)), axis=0) for game in games]

        return (np.asarray(data) for data in zip(*transitions))

    def __len__(self):
        return len(self.memory)


def load_replays(checkpoint_dir, memory_capacity):
    pickles = sorted(os.listdir(checkpoint_dir), reverse=True)
    pickles_paths = [os.path.join(checkpoint_dir, picke_file_name) for picke_file_name in pickles[:memory_capacity]]

    loaded_memory = []

    for picked_memory in pickles_paths:
        with open(picked_memory, 'rb') as file:
            loaded_memory += pickle.load(file)
        if len(loaded_memory) > memory_capacity:
            break

    loaded_memory = list(reversed(loaded_memory[:memory_capacity]))
    position = len(loaded_memory) % memory_capacity

    return loaded_memory, position


class Scheduler(object):
    def __init__(self, v, n_values, schedule):
        self.n = 0.
        self.v = v
        self.n_values = n_values
        self.schedule = schedules[schedule]

    def value(self):
        current_value = self.v * self.schedule(self.n / self.n_values)
        self.n += 1.
        return current_value

    def value_steps(self, steps):
        return self.v * self.schedule(steps / self.n_values)


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def mse(pred, target):
    return tf.square(pred - target) / 2.


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary
