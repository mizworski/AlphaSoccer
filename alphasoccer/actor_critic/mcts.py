import copy

import numpy as np
from recordclass import recordclass

from alphasoccer.environment.PaperSoccer import Soccer

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

debug=False

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
            a: (np.sqrt(N_sum) / (1 + self.transitions[a].N)) * self.c_puct * self.transitions[a].P
            for a in self.transitions
        }

        Q_U = {a: self.transitions[a].Q + U[a] for a in self.transitions}

        if debug:
            N = {a: self.transitions[a].N for a in self.transitions}
            Q = {a: self.transitions[a].Q for a in self.transitions}
            P = {a: self.transitions[a].P for a in self.transitions}

            print("N={}".format(N))
            print("Q={}".format(Q))
            print("P={}".format(P))
            print("U={}".format(U))
            print("v={}".format(self.v))

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
        probs = np.squeeze(probs)
        value = value.item()

        self.root = StateNode(probs, value, player=starting_player)
        assert self.envs[starting_player].get_player_turn() == 0
        assert self.envs[1 - starting_player].get_player_turn() == 1

    def rollout(self):
        rollout_envs = [copy.deepcopy(env) for env in self.envs]

        action, reward, done, parent_state_node, actions_history = traverse_tree(self.root, rollout_envs)

        player_turn_before_action = parent_state_node.player
        player_turn_after_action = rollout_envs[0].get_player_turn()

        if done:
            value = reward
            new_state_node = StateNode([], value, player_turn_before_action, legal_actions=[], terminal_state=True)
            # game has terminated, player in terminal state is set as one who made move to propagate reward properly
            player_turn_after_action = player_turn_before_action
        else:
            state = np.expand_dims(rollout_envs[player_turn_after_action].board.state, axis=0)
            probs, value = self.model.step(state)
            probs = np.squeeze(probs)
            value = value.item()

            if debug:
                print("End of rollout, last state.")
                print("Probs={}".format(probs))
                print("Value={}".format(value))
                print(rollout_envs[player_turn_after_action].board)
                print("player={}".format(player_turn_after_action))
                input()

            legal_actions = rollout_envs[player_turn_after_action].get_legal_moves()
            new_state_node = StateNode(probs, value, player=player_turn_after_action, legal_actions=legal_actions,
                                       c_puct=self.c_puct)

        parent_state_node.transitions[action].state_node = new_state_node

        # backup
        values = [None, None]
        values[player_turn_after_action] = value
        values[1 - player_turn_after_action] = -value

        self.root.backup(actions_history, values)

    def select_action(self, player, temperature=1):
        player_turn = self.envs[0].get_player_turn()
        assert player == player_turn

        for i in range(self.n_rollouts):
            self.rollout()

        action, pi = get_action_distribution(self.root.transitions, temperature)

        if debug:
            N = {a: self.root.transitions[a].N for a in self.root.transitions}
            Q = {a: self.root.transitions[a].Q for a in self.root.transitions}
            P = {a: self.root.transitions[a].P for a in self.root.transitions}

            print("N={}".format(N))
            print("Q={}".format(Q))
            print("P={}".format(P))
            print("v={}".format(self.root.v))

        return action, pi

    def step(self, action):
        if action not in self.root.transitions or self.root.transitions[action].state_node is None:
            # print('Node you are trying to reach is empty.')
            # warning: assuming action already has been made on envs
            player_turn = self.envs[0].get_player_turn()

            state = np.expand_dims(self.envs[player_turn].board.state, axis=0)
            probs, value = self.model.step(state)
            probs = np.squeeze(probs)
            value = value.item()
            legal_actions = self.envs[player_turn].get_legal_moves()

            terminal_state = np.sum(legal_actions) == 0

            self.root = StateNode(probs, value, player=player_turn, legal_actions=legal_actions,
                                  terminal_state=terminal_state)
        else:
            self.root = self.root.transitions[action].state_node


def traverse_tree(state_node, envs):
    action = None
    actions_history = []
    reward = None
    done = False
    parent_state_node = state_node

    # traversing game tree
    while state_node is not None and not done:
        action = state_node.select_next_action()
        player_taking_action = state_node.player

        if debug:
            print("player={}".format(player_taking_action))
            print("env")
            print(envs[player_taking_action])
            input()

        parent_state_node = state_node
        state_node = parent_state_node.transitions[action].state_node

        _, reward, done = envs[player_taking_action].step(action)
        _ = envs[1 - player_taking_action].step((action + 4) % 8)
        actions_history.append(action)

        if not (state_node is None or done == state_node.terminal_state):
            print('Game is done but state node is not terminal.')

    return action, reward, done, parent_state_node, actions_history
