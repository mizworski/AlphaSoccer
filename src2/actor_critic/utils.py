import random
import tensorflow as tf
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward'))

schedules = {
    'constant': lambda p: 1,
    'linear': lambda p: 1 - p
}


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def mse(pred, target):
    return tf.square(pred - target) / 2.


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Scheduler(object):
    def __init__(self, v, nvalues, schedule):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues
        self.schedule = schedules[schedule]

    def value(self):
        current_value = self.v * self.schedule(self.n / self.nvalues)
        self.n += 1.
        return current_value

    def value_steps(self, steps):
        return self.v * self.schedule(steps / self.nvalues)
