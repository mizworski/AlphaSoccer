import os
import sys
import re
import gc
import numpy as np
import pandas as pd
from multiprocessing import Process

parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent)

# from src.data.state_action_generate import StateActionGenerator
from src.soccer.game import Game

elo_threshold = 1985


def extract_game_details(lines):
    games_elos = []
    moves_sequences = []
    game_moves = ''
    white = -1
    black = -1

    for line in lines:
        if line == '' and game_moves != '':
            if moves_sequences != '1. ' and white > elo_threshold and black > elo_threshold:
                moves_sequences.append(game_moves)
                games_elos.append((white, black))

            game_moves = ''
            white = -1
            black = -1
        elif not line.startswith('['):
            game_moves += line
        elif 'WhiteElo' in line:
            white = int(re.findall('"([^"]*)"', line)[0])
        elif 'BlackElo' in line:
            black = int(re.findall('"([^"]*)"', line)[0])

    if game_moves != '' and white > elo_threshold and black > elo_threshold:
        moves_sequences.append(game_moves)
        games_elos.append((white, black))

    games_moves = []
    for moves in moves_sequences:
        history = []
        for seq in moves.split()[:-1]:
            if '.' not in seq:
                moves = list(seq)
                history.append(moves)

        games_moves.append(history)

    return games_elos, games_moves


def generate_state_actions(games_elos, games_moves, games_to_process, thread_id):
    games_processed = 0
    for elos, turns in zip(games_elos, games_moves):
        game = Game(elos[0], elos[1])
        state_actions_game = []
        for i, turn in enumerate(turns):
            for move in turn:
                action = int(move)
                state = game.boards[i % 2].board
                game.make_move(action)
                state_action = np.append(state.flatten(), [action])
                state_actions_game.append(state_action)

        games_processed += 1
        if len(state_actions_game) != 0:
            to_save = np.array(state_actions_game, dtype=np.int8)
            formatter = '%d'
            np.savetxt('data/games/{}.csv'.format(games_processed + games_to_process * thread_id),
                       to_save, delimiter=',', fmt=formatter)
        if games_processed == games_to_process:
            break


if __name__ == '__main__':
    fp = open('data/raw/sc.txt')
    lines = fp.read().split("\n")
    fp.close()

    games_elos, games_moves = extract_game_details(lines)

    print("Elos: {}".format(len(games_elos)))
    print("Moves: {}".format(len(games_moves)))

    # games_per_thread = int(len(games_elos) / 8)
    games_per_thread = int(len(games_moves) / 8)
    # games_per_thread = 50

    ps = []
    print('Starting processes')
    for i in range(8):
        args = (games_elos[i * games_per_thread:(i + 1) * games_per_thread],
                games_moves[i * games_per_thread:(i + 1) * games_per_thread],
                games_per_thread, i)
        p = Process(target=generate_state_actions, args=args)
        p.start()
        ps.append(p)

    print('Joining')
    for i in range(8):
        ps[i].join()
