import numpy as np

from src2.environment.PaperSoccer import Soccer
from src2.actor_critic.model import Model
from src2.actor_critic.utils import ReplayMemory


def select_action(probs, legal_moves, n_act, temperature):
    probs = [prob ** (1 / temperature) for prob in probs]
    probs_only_legal = np.multiply(probs, legal_moves)
    l1_norm = np.linalg.norm(probs_only_legal, ord=1)
    if l1_norm == 0:
        probs_fixed = probs / np.linalg.norm(probs, ord=1)
    else:
        probs_fixed = probs_only_legal / l1_norm

    return np.random.choice(range(n_act), p=probs_fixed)


class Runner(object):
    def __init__(self, initial_model, n_replays):
        self.envs = [Soccer(), Soccer()]
        self.best_player = initial_model
        self.replay_memory = ReplayMemory(n_replays)

    def run(self, n_games, temperature=1):
        n_act = Soccer.action_space.n

        for _ in range(n_games):
            states = [self.envs[i].reset(i) for i in range(2)]

            history = [[], []]

            done = False
            while not done:
                player_turn = self.envs[0].get_player_turn()
                probs, _ = self.best_player.step(states[player_turn])
                legal_moves = self.envs[player_turn].get_legal_moves()
                action = select_action(probs[0], legal_moves, n_act, temperature)
                state, reward, done = self.envs[player_turn].step(action)

                action_opposite_player_perspective = (action + 4) % 8
                state_opp, _, _ = self.envs[1 - player_turn].step(action_opposite_player_perspective)

                history[player_turn].append([state, action])

            for state, action in history[player_turn]:
                self.replay_memory.push(state, action, reward)
            for state, action in history[1 - player_turn]:
                self.replay_memory.push(state, action, -reward)


def main():
    n_batch = 1024
    model = Model(Soccer.observation_space, Soccer.action_space, n_batch)

    r = Runner(model, 10)
    r.run(1024, temperature=2)


if __name__ == '__main__':
    main()
