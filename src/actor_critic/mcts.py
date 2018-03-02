import os
import copy
from time import sleep

import numpy as np
from recordclass import recordclass

from src.actor_critic.model import Model
from src.environment.PaperSoccer import Soccer

ActionStatistics = recordclass('ActionStatistics', ('state_node', 'N', 'W', 'Q', 'P'))


def get_action_distribution(transitions, temperature, verbose=0):
    actions = list(range(Soccer.action_space.n))
    Ns = [
        transitions[action].N
        if action in transitions
        else 0
        for action in range(Soccer.action_space.n)
    ]
    NsT = [N ** (1 / temperature) for N in Ns]
    pi = [N / np.sum(NsT) for N in NsT]
    action = np.random.choice(actions, p=pi)

    if verbose:
        print(Ns)
        print('action={}'.format(action))
        print(16 * '*')

    return action, pi


class StateNode:
    def __init__(self, probs, value, player, legal_actions=None, terminal_state=False, c_puct=1):
        self.v = value
        # todo probs don't sum up to 1 due to legal actions clip
        if legal_actions is None:
            legal_actions = [1] * Soccer.action_space.n
        self.transitions = {
            action: ActionStatistics(state_node=None, N=0, W=0, Q=0, P=probs[action])
            for action in range(len(legal_actions)) if legal_actions[action] == 1
        }

        if len(self.transitions) == 0 and not terminal_state:
            print('cc {}'.format(terminal_state))

        self.player = player
        self.terminal_state = terminal_state
        self.c_puct = c_puct

    def select_next_action(self):
        N_sum = np.sum([self.transitions[i].N for i in self.transitions])
        U = {
            a: (1 + np.sqrt(N_sum) / (1 + self.transitions[a].N)) * self.c_puct * self.transitions[a].P
            for a in self.transitions
        }

        Q_U = {a: self.transitions[a].Q + U[a] for a in self.transitions}

        action = max(Q_U, key=Q_U.get)

        return action

    def backup(self, actions, values, verbose=0):
        if verbose:
            print('backing up state')
            print('player={}'.format(self.player))
            print('value={}'.format(values[self.player]))

        action = actions[0]

        self.transitions[action].N += 1
        self.transitions[action].W += values[self.player]
        self.transitions[action].Q = self.transitions[action].W / self.transitions[action].N

        if len(actions[1:]) > 0:
            self.transitions[action].state_node.backup(actions[1:], values)


class MCTS:
    def __init__(self, envs, model, n_rollouts=1600, c_puct=1):
        self.envs = envs
        self.model = model
        self.root = None
        # self.temperature = temperature
        self.n_rollouts = n_rollouts
        self.c_puct = c_puct

    def reset(self, starting_player):
        state = np.expand_dims(self.envs[starting_player].board.state, axis=0)
        probs, value = self.model.step(state)
        probs, value = np.squeeze(probs), np.squeeze(value)
        self.root = StateNode(probs, value, player=starting_player)
        assert self.envs[starting_player].get_player_turn() == 0
        assert self.envs[1 - starting_player].get_player_turn() == 1

    def rollout(self):
        rollout_envs = [copy.deepcopy(env) for env in self.envs]
        tree_state_node = self.root
        parent_tree_state_node = tree_state_node

        action = None
        actions_history = []
        reward = None
        done = False

        # traversing game tree
        while tree_state_node is not None and not done:
            action = tree_state_node.select_next_action()
            player_taking_action = tree_state_node.player
            parent_tree_state_node = tree_state_node
            tree_state_node = parent_tree_state_node.transitions[action].state_node

            _, reward, done = rollout_envs[player_taking_action].step(action)
            _ = rollout_envs[1 - player_taking_action].step((action + 4) % 8)
            actions_history.append(action)

            if not (tree_state_node is None or done == tree_state_node.terminal_state):
                print('Game is done but state node is not terminal.')

        # expand and evaluate
        new_state_player_turn = rollout_envs[0].get_player_turn()
        last_player_turn = parent_tree_state_node.player

        if done:
            value = reward
            tree_state_node = StateNode([], value, last_player_turn, legal_actions=[], terminal_state=True)
            # sleep(3)
        else:
            state = np.expand_dims(rollout_envs[new_state_player_turn].board.state, axis=0)
            probs, value = self.model.step(state)
            probs, value = np.squeeze(probs), np.squeeze(value)

            legal_actions = rollout_envs[new_state_player_turn].get_legal_moves()
            # legal_actions = [i for i, val in enumerate(legal_actions_sparse) if val == 1]

            tree_state_node = StateNode(probs, value, player=new_state_player_turn, legal_actions=legal_actions,
                                        c_puct=self.c_puct)

        parent_tree_state_node.transitions[action].state_node = tree_state_node

        # backup
        values = [None, None]
        values[new_state_player_turn] = value
        values[1 - new_state_player_turn] = -value

        self.root.backup(actions_history, values)

    def select_action(self, player, temperature=1):
        player_turn = self.envs[0].get_player_turn()
        assert player == player_turn

        for i in range(self.n_rollouts):
            self.rollout()

        action, pi = get_action_distribution(self.root.transitions, temperature)

        return action, pi

    def step(self, action):
        if action not in self.root.transitions or self.root.transitions[action].state_node is None:
            # print('Node you are trying to reach is empty.')
            # warning: assuming action already has been made on envs
            player_turn = self.envs[0].get_player_turn()

            state = np.expand_dims(self.envs[player_turn].board.state, axis=0)
            probs, value = self.model.step(state)
            probs, value = np.squeeze(probs), np.squeeze(value)
            legal_actions = self.envs[player_turn].get_legal_moves()

            terminal_state = np.sum(legal_actions) == 0

            self.root = StateNode(probs, value, player=player_turn, legal_actions=legal_actions,
                                  terminal_state=terminal_state)
        else:
            self.root = self.root.transitions[action].state_node
