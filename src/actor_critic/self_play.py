import numpy as np

from src.actor_critic.mcts import MCTS
from src.actor_critic.utils import ReplayMemory
from src.environment.PaperSoccer import Soccer


def playing_progress_bar(game, n_games):
    log_every_n_games = max(2, n_games // 10)
    if (game + 1) % log_every_n_games == 0:
        print("Completed {}% of self-play.".format(int(100 * (game + 1) / n_games)))


class Runner(object):
    def __init__(self, initial_model, n_replays, c_puct, replay_checkpoint_dir, checkpoint_every_n_transitions,
                 verbose=0):
        self.envs = [Soccer(), Soccer()]
        self.best_player = initial_model
        self.replay_memory = ReplayMemory(n_replays, replay_checkpoint_dir=replay_checkpoint_dir,
                                          checkpoint_every_n_transitions=checkpoint_every_n_transitions,
                                          verbose=verbose)
        self.c_puct = c_puct

    def run(self, n_games=int(1e4), initial_temperature=1.0, n_rollouts=1600, temperature_decay_factor=0.95,
            moves_before_dacaying=8):
        mcts = [
            MCTS(self.envs, self.best_player, n_rollouts=n_rollouts, c_puct=self.c_puct)
            for _ in range(2)
        ]

        for game in range(n_games):
            winner, history = play_single_game(self.envs, mcts, initial_temperature=initial_temperature,
                                               starting_player=0, temperature_decay_factor=temperature_decay_factor,
                                               moves_before_dacaying=moves_before_dacaying)
            save_memory(self.replay_memory, winner, history)
            playing_progress_bar(game, n_games)

    def evaluate(self, model, n_games=400, initial_temperature=0.25, n_rollouts=1600, new_best_model_threshold=0.55,
                 temperature_decay_factor=0.95, moves_before_dacaying=8, verbose=0):
        log_every_n_games = max(2, n_games // 4)
        n_wins = 0
        mcts = [
            MCTS(self.envs, model, n_rollouts=n_rollouts),
            MCTS(self.envs, self.best_player, n_rollouts=n_rollouts)
        ]

        for game in range(n_games):
            starting_player = game % 2
            print_results = verbose if verbose and game % log_every_n_games == 0 else 0

            winner, _ = play_single_game(self.envs, mcts, starting_player=starting_player,
                                         initial_temperature=initial_temperature,
                                         temperature_decay_factor=temperature_decay_factor,
                                         moves_before_dacaying=moves_before_dacaying, verbose=print_results)

            if winner == 0:
                n_wins += 1

        if verbose:
            print("Win ratio of new model = {:.2f}".format(100 * n_wins / n_games))

        if n_wins / n_games > new_best_model_threshold:
            self.best_player = model
            return True
        else:
            return False


def play_single_game(envs, mcts, starting_player=0, initial_temperature=1.0,
                     temperature_decay_factor=0.95, moves_before_dacaying=8, verbose=0):
    history = [[], []]

    for i in range(2):
        # is player0 starts then env1 has to set starting player as player1 (cause real player1 is not starting)
        starting_from_i_perspective = abs(i - starting_player)
        envs[i].reset(starting_game=starting_from_i_perspective)
    for i in range(2):
        mcts[i].reset(starting_player=starting_player)

    done = False
    player_turn = None
    reward = None
    temperature = initial_temperature

    moves = 0
    while not done:
        if verbose == 2:
            envs[0].print_board()

        # env0 indicates player turn (env1 is reversed)
        player_turn = envs[0].get_player_turn()
        state = envs[player_turn].board.state
        action, pi = mcts[player_turn].select_action(player_turn, temperature=temperature)
        action_opposite_player_perspective = (action + 4) % 8

        _, reward, done = envs[player_turn].step(action)
        _ = envs[1 - player_turn].step(action_opposite_player_perspective)

        mcts[player_turn].step(action)
        mcts[1 - player_turn].step(action)

        if history is not None:
            history[player_turn].append([np.squeeze(state.copy()), pi])

        moves += 1
        if moves > moves_before_dacaying and temperature > 5e-2:
            temperature *= temperature_decay_factor

    if (player_turn == 0 and reward > 0) or (player_turn == 1 and reward < 0):
        winner = 0
    else:
        winner = 1

    if verbose:
        envs[0].print_board()
        if winner == 0:
            print("Player won")
        else:
            print("Player lost")

    return winner, history


def save_memory(replay_memory, winner, history):
    for state, pi in history[winner]:
        replay_memory.push(state, pi, 1)
    for state, pi in history[1 - winner]:
        replay_memory.push(state, pi, -1)
