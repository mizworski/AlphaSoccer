import os
import sys
import pickle
from collections import namedtuple
from datetime import datetime

import numpy as np
import tensorflow as tf

Transition = namedtuple('Transition', ('state', 'pi', 'reward'))

schedules = {
    'constant': lambda p: 1,
    'linear': lambda p: 1 - p,
    'stairs': lambda p: 10 ** -int(p * 3)
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
            print("Loaded {} games from {}".format(len(self.memory), self.replay_checkpoint_dir), file=sys.stderr)

    def push_vector(self, sars):
        transitions = [Transition(*sar) for sar in sars]

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transitions

        if self.verbose == 2:
            print("Transitions in history = {}".format(len(transitions)), file=sys.stderr)

            i = 0
            for transition in self.memory[self.position]:
                print("Transition {}".format(i))
                print("Reward : {}".format(transition.reward))
                print("Pi: {}".format(transition.pi))
                print("State")
                print(transition.state)
                print("*" * 8)
                i += 1
                if i >= 8:
                    break

        self.position = (self.position + 1) % self.capacity

        if self.position % self.n_games_in_replay_checkpoint == 0:
            time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            output_file_name = 'checkpoint_{}_{}.pickle'.format(time_str, self.position)
            output_file_path = os.path.join(self.replay_checkpoint_dir, output_file_name)

            if self.position == 0:
                memory_slice = self.memory[-self.n_games_in_replay_checkpoint:]
            else:
                memory_slice = self.memory[self.position - self.n_games_in_replay_checkpoint:self.position]

            if output_file_path.startswith('gs://'):
                with tf.gfile.GFile(output_file_path, 'wb') as file:
                    pickle.dump(memory_slice, file)
            else:
                with open(output_file_path, 'wb') as file:
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
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    pickles = sorted(os.listdir(checkpoint_dir), reverse=True)
    pickles_paths = [os.path.join(checkpoint_dir, picke_file_name) for picke_file_name in pickles[:memory_capacity]]

    loaded_memory = []

    for pickled_memory in pickles_paths:
        if pickled_memory.startswith('gs://'):
            with tf.gfile.GFile(pickled_memory, 'wb') as file:
                loaded_memory += pickle.load(file)
        else:
            with open(pickled_memory, 'rb') as file:
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

    def reset_steps(self):
        self.n = 0
