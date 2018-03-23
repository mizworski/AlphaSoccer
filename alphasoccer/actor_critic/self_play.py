from multiprocessing.pool import ThreadPool

import numpy as np
import tensorflow as tf
import tqdm

from alphasoccer.actor_critic.mcts import MCTS
from alphasoccer.actor_critic.utils import ReplayMemory
from alphasoccer.environment.PaperSoccer import Soccer


class Runner(object):
    def __init__(self, initial_model, n_replays, c_puct, replay_checkpoint_dir, n_games_in_replay_checkpoint,
                 verbose=0):
        self.model = initial_model
        self.replay_memory = ReplayMemory(n_replays, replay_checkpoint_dir=replay_checkpoint_dir,
                                          n_games_in_replay_checkpoint=n_games_in_replay_checkpoint,
                                          verbose=verbose)
        self.c_puct = c_puct

        self.n_wins_ph = tf.placeholder(tf.int32)
        self.n_games_ph = tf.placeholder(tf.int32)
        self.win_ratio_summary_op = tf.summary.scalar('win_ratio', 100 * self.n_wins_ph / self.n_games_ph)

    def run(self, n_games=int(1e4), initial_temperature=1.0, n_rollouts=1600, temperature_decay_factor=0.95,
            moves_before_dacaying=8, verbose=0):
        pool = ThreadPool()
        progress_bar = tqdm.tqdm(total=n_games)

        assert verbose == 0

        arguments = (
            self.model.step_model, self.model.step_model, n_rollouts, self.c_puct, 0, initial_temperature,
            temperature_decay_factor, moves_before_dacaying, progress_bar, None, None, None, None, verbose
        )
        iterable_arguments = [arguments] * n_games

        res = pool.starmap_async(play_single_game, iterable_arguments)
        res.wait()

        progress_bar.close()

        for (winner, history) in res.get():
            save_memory(self.replay_memory, winner, history)

    def evaluate(self, model, n_games=400, initial_temperature=0.25, n_rollouts=1600, new_best_model_threshold=0.55,
                 temperature_decay_factor=0.95, moves_before_dacaying=8, eval_iter=0, verbose=0):
        # log_every_n_games = max(2, n_games // 4)
        log_n_games = 8
        n_wins = 0

        pool = ThreadPool()
        progress_bar = tqdm.tqdm(total=n_games)

        iterable_arguments = [
            (
                self.model.train_model, self.model.step_model, n_rollouts, self.c_puct, (i % 2), initial_temperature,
                temperature_decay_factor, moves_before_dacaying, progress_bar,
                eval_iter, i, model.sess, model.summary_writer,
                # verbose if verbose and i % log_every_n_games == 0 else 0
                1 if i < log_n_games else 0
            )
            for i in range(n_games)
        ]

        res = pool.starmap_async(play_single_game, iterable_arguments)
        res.wait()

        progress_bar.close()

        for (winner, _) in res.get():
            if winner == 0:
                n_wins += 1

        win_ratio_summary = model.sess.run(self.win_ratio_summary_op,
                                           feed_dict={self.n_wins_ph: n_wins, self.n_games_ph: n_games})
        model.summary_writer.add_summary(win_ratio_summary, eval_iter)

        return n_wins / n_games > new_best_model_threshold


def play_single_game(model0, model1=None, n_rollouts=800, c_puct=1, starting_player=0, initial_temperature=1.0,
                     temperature_decay_factor=0.95, moves_before_dacaying=8, progress_bar=None, epoch_iter=0,
                     game_iter=0, sess=None, summary_writer=None, verbose=0):
    if model1 is None:
        model1 = model0
    envs = [Soccer(), Soccer()]
    mcts = [
        MCTS(envs, model0, n_rollouts=n_rollouts, c_puct=c_puct),
        MCTS(envs, model1, n_rollouts=n_rollouts, c_puct=c_puct)
    ]
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
        board_str = str(envs[0].board)
        text_tensor = tf.make_tensor_proto(board_str, dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"

        summary = tf.Summary()
        summary_tag = 'eval_{}'.format(epoch_iter)
        summary.value.add(tag=summary_tag, metadata=meta, tensor=text_tensor)

        # summary_op = tf.summary.text(, tf.convert_to_tensor(board_str))
        # text = sess.run(summary_op)
        summary_writer.add_summary(summary, game_iter)

    if verbose:
        envs[0].print_board()
        if winner == 0:
            print("Player won")
        else:
            print("Player lost")

    progress_bar.update(1)
    return winner, history


def save_memory(replay_memory, winner, history):
    sars = []

    for state, pi in history[winner]:
        sars.append((state, pi, 1))
    for state, pi in history[1 - winner]:
        sars.append((state, pi, -1))

    replay_memory.push_vector(sars)
