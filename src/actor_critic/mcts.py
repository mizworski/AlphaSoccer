import os
import copy
from time import sleep

import numpy as np
from recordclass import recordclass

from src.actor_critic.model import Model
from src.environment.PaperSoccer import Soccer

ActionStatistics = recordclass('ActionStatistics', ('state_node', 'N', 'W', 'Q', 'P'))


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
    def __init__(self, envs, model, temperature, n_rollouts=1600, c_puct=1):
        self.envs = envs
        self.model = model
        self.temperature = temperature
        self.n_rollouts = n_rollouts
        self.tree = MctsTree(envs, model, temperature, n_rollouts=n_rollouts, c_puct=c_puct)

    def reset(self, player_number):
        self.tree.reset(player_number)

    def select_action(self, player):
        action, pi = self.tree.select_action(player)

        # be careful here - it assumes you take this action afterward
        # self.tree.play(action)
        #
        # if player == 1:
        #     action = (action + 4) % Soccer.action_space.n
        #     pi = rotate_probabilities(pi)

        return action, pi

    def step(self, action):
        self.tree.play(action)


def rotate_probabilities(probs):
    return [probs[(i + 4) % Soccer.action_space.n] for i in range(8)]


class StateNode:
    # n_act = Soccer.action_space.n

    def __init__(self, probs, value, player, legal_actions=None, terminal_state=False, c_puct=1):
        self.v = value
        # todo probs don't sum up to 1 due to legal actions clip
        if legal_actions is None:
            legal_actions = range(Soccer.action_space.n)
        self.transitions = {
            action: ActionStatistics(state_node=None, N=0, W=0, Q=0, P=probs[action])
            for action in legal_actions
        }
        self.player = player
        self.terminal_state = terminal_state
        self.c_puct = c_puct

    def select(self):
        N_sum = np.sum([self.transitions[i].N for i in self.transitions])
        U = {
            a: (1 + np.sqrt(N_sum) / (1 + self.transitions[a].N)) * self.c_puct * self.transitions[a].P
            for a in self.transitions
        }

        Q_U = {a: self.transitions[a].Q + U[a] for a in self.transitions}

        action = max(Q_U, key=Q_U.get)
        # try:
        # except:
        #     print('dupa')
        #     action = 0

        return action

    def backup(self, actions, values):
        # todo why passing list instead of value
        print('backing up state')
        print('player={}'.format(self.player))
        print('value={}'.format(values[self.player]))
        action = actions[0]
        self.transitions[action].N += 1
        self.transitions[action].W += values[self.player]
        self.transitions[action].Q = self.transitions[action].W / self.transitions[action].N
        if len(actions[1:]) > 0:
            self.transitions[action].state_node.backup(actions[1:], values)


class MctsTree:
    def __init__(self, envs, model, initial_temperature, n_rollouts=1600, c_puct=1):
        self.envs = envs
        self.model = model
        self.root = None
        self.temperature = initial_temperature
        self.n_rollouts = n_rollouts
        self.c_puct = c_puct
        self.player_number = None

    def reset(self, player_number):
        state = np.expand_dims(self.envs[0].board.state, axis=0)
        probs, value = self.model.step(state)
        probs, value = np.squeeze(probs), np.squeeze(value)
        self.root = StateNode(probs, value, player=0)
        self.player_number = player_number
        assert self.envs[0].get_player_turn() == 0

    def rollout(self):
        if self.player_number == 0:
            rollout_envs = [copy.deepcopy(env) for env in self.envs]
        else:
            rollout_envs = [copy.deepcopy(env) for env in reversed(self.envs)]

        tree_state_node = self.root
        parent_tree_state_node = tree_state_node

        action = None
        actions_history = []
        reward = None
        done = False

        # traversing game tree
        while tree_state_node is not None and not done:
            action = tree_state_node.select()
            parent_tree_state_node = tree_state_node
            tree_state_node = parent_tree_state_node.transitions[action].state_node

            _, reward, done = rollout_envs[0].step(action)
            _ = rollout_envs[1].step((action + 4) % 8)
            actions_history.append(action)

        # expand and evaluate
        new_state_player_turn = rollout_envs[0].get_player_turn()

        if done:
            print('game finished, reward={}'.format(reward))
            value = reward
            player_reaching_state = parent_tree_state_node.player
            tree_state_node = StateNode([], value, player_reaching_state, [], terminal_state=True)
        else:
            # print("last player={}".format(new_state_player_turn))
            print("player={}".format(self.player_number))
            rollout_envs[0].print_board()
            sleep(1)

            state = np.expand_dims(rollout_envs[new_state_player_turn].board.state, axis=0)
            probs, value = self.model.step(state)
            probs, value = np.squeeze(probs), np.squeeze(value)

            if new_state_player_turn == 1:  # opponent
                probs = rotate_probabilities(probs)
                value = -value

            legal_actions_sparse = rollout_envs[0].get_legal_moves()
            legal_actions = [i for i, val in enumerate(legal_actions_sparse) if val == 1]
            print('legal actions:')
            print(legal_actions)
            tree_state_node = StateNode(probs, value, player=new_state_player_turn, legal_actions=legal_actions,
                                        c_puct=self.c_puct)

        parent_tree_state_node.transitions[action].state_node = tree_state_node

        # backup
        values = [None, None]
        values[new_state_player_turn] = value
        values[1 - new_state_player_turn] = -value

        self.root.backup(actions_history, values)

    def select_action(self, player):
        player_turn = self.envs[0].get_player_turn()
        assert player == player_turn

        for i in range(self.n_rollouts):
            print(16 * '*')
            print('rollout={}'.format(i))
            self.rollout()

        actions = list(range(Soccer.action_space.n))
        Ns = [
            self.root.transitions[action].N
            if action in self.root.transitions
            else 0
            for action in range(Soccer.action_space.n)
        ]
        print(Ns)
        NsT = [N ** (1 / self.temperature) for N in Ns]
        pi = [N / np.sum(NsT) for N in NsT]
        action = np.random.choice(actions, p=pi)
        print('action={}'.format(action))
        print(16 * '*')
        return action, pi

    def play(self, action):
        if self.root.transitions[action].state_node is None:
            print('Node you are trying to reach is empty.')
            state = np.expand_dims(self.envs[0].board.state, axis=0)
            probs, value = self.model.step(state)
            probs, value = np.squeeze(probs), np.squeeze(value)

            player_turn = self.envs[0].get_player_turn()
            legal_actions = self.envs[0].get_legal_moves()
            print(legal_actions)
            self.root = StateNode(probs, value, player=player_turn, legal_actions=legal_actions)
        else:
            self.root = self.root.transitions[action].state_node


def main():
    envs = [Soccer(), Soccer()]
    model_dir = os.path.join('models', 'actor_critic')
    temperature = 0.5

    model = Model(Soccer.observation_space, Soccer.action_space, batch_size=None, lr=None,
                  training_timesteps=None, model_dir=model_dir, verbose=None)

    tree = MctsTree(envs, model, temperature, n_rollouts=800)
    for i in range(2):
        envs[i].reset(starting_game=i)

    tree.reset(0)

    tree.select_action(0)


if __name__ == '__main__':
    main()
