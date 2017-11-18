import os
import copy
import numpy as np
from collections import namedtuple

from src.actor_critic.model import Model
from src.environment.PaperSoccer import Soccer

ActionStatistics = namedtuple('ActionStatistics', ('state_node', 'N', 'W', 'Q', 'P'))


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
    def __init__(self, envs, model, temperature, n_rollouts=1600):
        self.envs = envs
        self.model = model
        self.temperature = temperature
        self.n_rollouts = n_rollouts

    def select_action(self, player):
        player_turn = self.envs[0].get_player_turn()
        assert player == player_turn
        n_act = Soccer.action_space.n

        state = np.expand_dims(self.envs[player].board.state, axis=0)
        probs, value = self.model.step(state)
        legal_moves = self.envs[player].get_legal_moves()

        action = select_action(np.squeeze(probs), legal_moves, n_act, temperature=self.temperature)

        return action


def rotate_probabilities(probs):
    return [probs[(i + 4) % Soccer.action_space.n] for i in range(8)]


class StateNode:
    n_act = Soccer.action_space.n

    def __init__(self, probs, value, player, legal_actions=range(n_act), previous_state=None, c_puct=1):
        self.v = value
        self.transition = {
            action: ActionStatistics(state_node=None, N=0, W=0, Q=0, P=probs[action])
            for action in range(8)
        }
        self.player = player
        self.c_puct = c_puct
        # self.previous_state = previous_state

    def select(self):
        N_sum = np.sum([self.transition[i].N for i in self.transition])
        U = {
            a: (1 + np.sqrt(N_sum) / (1 + self.transition[a].N)) * self.c_puct * self.transition[a].P
            for a in self.transition
        }

        Q_U = {a: self.transition[a].Q + U[a] for a in self.transition}

        action = max(Q_U, key=Q_U.get)
        return action

    def backup(self, actions, values):
        if len(actions) > 0:
            action = actions[0]
            self.transition[action].N += 1
            self.transition[action].W += values[self.player]
            self.transition[action].state_node.backup(actions[1:], values)


class MctsTree:
    def __init__(self, envs, model, initial_temperature, c_puct=1):
        self.envs = envs
        self.model = model
        self.root_nodes = None
        self.temperature = initial_temperature
        self.c_puct = c_puct

    def reset(self):
        state = np.expand_dims(self.envs[0].board.state, axis=0)
        probs, value = self.model.step(state)
        probs, value = np.squeeze(probs), np.squeeze(value)
        self.root_nodes = [StateNode(probs, value, player=0), StateNode(rotate_probabilities(probs), value, player=0)]

    def rollout(self):
        rollout_envs = [copy.deepcopy(env) for env in self.envs]

        tree_state_nodes = self.root_nodes
        parent_tree_state_nodes = tree_state_nodes
        actions = [-1, -1]

        # traversing game tree
        actions_history = []
        while tree_state_nodes != [None, None]:
            actions = [-1, -1]
            for i in range(2):
                actions[i] = tree_state_nodes[i].select()
                parent_tree_state_nodes[i] = copy.copy(tree_state_nodes[i])
                tree_state_nodes[i] = parent_tree_state_nodes[i].transition[actions[i]].state_node
                rollout_envs[i].step(actions[i])

            assert actions[0] == (actions[1] + 4) % 8
            actions_history.append(actions)

        # expand and evaluate
        leaf_nodes = parent_tree_state_nodes
        last_player_turn = rollout_envs[0].get_player_turn()
        state = np.expand_dims(rollout_envs[last_player_turn].board.state, axis=0)
        probs, value = self.model.step(state)
        probs, value = np.squeeze(probs), np.squeeze(value)
        probs_both_perspective = [probs, rotate_probabilities(probs)]

        for i in range(2):
            tree_state_nodes[i] = StateNode(probs_both_perspective[i], value, last_player_turn,
                                           rollout_envs[i].get_legal_moves(), leaf_nodes[i], self.c_puct)
            parent_tree_state_nodes[i].transition[actions[i]].state_node = tree_state_nodes[i]

        # backup
        values = [None, None]
        values[last_player_turn] = value
        values[1 - last_player_turn] = -value
        actions_history = zip(*actions_history)
        for i in range(2):
            self.root_nodes[i].backup(actions_history[i], values)

    def select_action(self, player):
        player_turn = self.envs[0].get_player_turn()
        assert player == player_turn


def main():
    # state = StateNode([0, 0.2, 0, 0, 0, 0.3, 0.4, 0, 0, 0.1], 0.2, 1)
    # act = state.select()
    # print(act)

    envs = [Soccer(), Soccer()]
    model_dir = os.path.join('models', 'actor_critic')
    temperature = 0.5

    model = Model(Soccer.observation_space, Soccer.action_space, batch_size=None, lr=None,
                  training_timesteps=None, model_dir=model_dir, verbose=None)

    tree = MctsTree(envs, model, temperature)
    for i in range(2):
        envs[i].reset(starting_game=i)

    tree.reset()
    tree.rollout()


if __name__ == '__main__':
    main()
