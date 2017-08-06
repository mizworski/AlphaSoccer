import numpy as np


class Agent:
    def __init__(self, length, width):
        self._directions = 8
        self.lookup_table = np.zeros((length + 1, width + 1, self._directions))
