import copy
import numpy as np
import tensorflow as tf

from src.environment.PaperSoccer import Soccer


def to_tuple(a):
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a


def select_action(probs, legal_moves, n_act, temperature=None):
    if temperature is None:
        probs_only_legal = np.multiply(probs, legal_moves)
        return np.argmax(probs_only_legal)

    probs = [prob ** (1 / temperature) for prob in probs]
    probs_only_legal = np.multiply(probs, legal_moves)
    l1_norm = np.linalg.norm(probs_only_legal, ord=1)
    if l1_norm == 0:
        probs_fixed = probs / np.linalg.norm(probs, ord=1)
    else:
        probs_fixed = probs_only_legal / l1_norm

    return np.random.choice(range(n_act), p=probs_fixed)


class MCTS:
    def __init__(self, env, model, temperature, n_rollouts=1024):
        self.env = env
        self.model = model
        self.temperature = temperature
        self.n_rollouts = n_rollouts

    def select_action(self):
        n_act = Soccer.action_space.n

        state_visit_count = {}
        state = np.expand_dims(self.env.board.state, axis=0)
        probs, value = self.model.step(state)
        legal_moves = self.env.get_legal_moves()

        action = select_action(probs[0], legal_moves, n_act, temperature=self.temperature)

        # for _ in range(self.n_rollouts):
        #     rollout_env = copy.deepcopy(self.env)
        return action, value



