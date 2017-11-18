import numpy as np

from src.actor_critic.mcts import MCTS
from src.environment.PaperSoccer import Soccer
from src.actor_critic.utils import ReplayMemory


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


class Runner(object):
    def __init__(self, initial_model, n_replays):
        self.envs = [Soccer(), Soccer()]
        self.best_player = initial_model
        self.replay_memory = ReplayMemory(n_replays)

    def run(self, n_games=int(1e4), temperature=1, n_rollouts=1600):
        mcts = MCTS(self.envs, self.best_player, temperature=temperature, n_rollouts=n_rollouts)

        for game in range(n_games):
            for i in range(2):
                self.envs[i].reset(i)
            mcts.reset()
            history = [[], []]

            done = False
            while not done:
                player_turn = self.envs[0].get_player_turn()
                state = self.envs[player_turn].board.state
                action, pi = mcts.select_action(player_turn)
                _, reward, done = self.envs[player_turn].step(action)

                action_opposite_player_perspective = (action + 4) % 8
                _ = self.envs[1 - player_turn].step(action_opposite_player_perspective)

                history[player_turn].append([np.squeeze(state), action,])

            for state, action in history[player_turn]:
                self.replay_memory.push(state, action, reward)
            for state, action in history[1 - player_turn]:
                self.replay_memory.push(state, action, -reward)

            if (game + 1) % (n_games // 10) == 0:
                print("Completed {}% of self-play.".format(int(100 * (game + 1) / n_games)))

    def evaluate(self, model, temperature=0.25, new_best_model_threshold=0.55, verbose=0):
        n_games = 400
        log_every_n_games = n_games // 4
        n_wins = 0
        mcts = [MCTS(self.envs, model, temperature=temperature),
                MCTS(self.envs, self.best_player, temperature=temperature)]

        for game in range(n_games):
            starting_player = game % 2
            for i in range(2):
                self.envs[i].reset(starting_game=abs(i - starting_player))

            done = False
            while not done:
                player_turn = self.envs[0].get_player_turn()
                action, _ = mcts[player_turn].select_action(player_turn)

                if verbose == 2 and game % log_every_n_games == 0:
                    self.envs[0].print_board()

                _, reward, done = self.envs[player_turn].step(action)

                action_opposite_player_perspective = (action + 4) % 8
                _ = self.envs[1 - player_turn].step(action_opposite_player_perspective)


            if verbose == 1 and game % log_every_n_games == 0:
                self.envs[0].print_board()

            if (player_turn == 0 and reward > 0) or (player_turn == 1 and reward < 0):
                n_wins += 1
                if verbose == 1 and game % log_every_n_games == 0:
                    print("Player won")
            elif verbose == 1 and game % log_every_n_games == 0:
                print("Player lost")

        if verbose == 1:
            print("Win ratio of new model = {:.2f}".format(100 * n_wins / n_games))

        if n_wins / n_games > new_best_model_threshold:
            self.best_player = model
            return True

        return False
