import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from soccer.board import Board
from soccer.game import Game

length = 10
width = 8


def extract_game_details(lines):
    games_elos = []
    moves_sequences = []
    game_moves = ''
    white = -1
    black = -1

    for line in lines:
        if line == '' and game_moves != '':
            if moves_sequences != '1. ':
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

    if game_moves != '':
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


def generate_state_actions(games_to_process):
    fp = open('data/sc.txt')
    lines = fp.read().split("\n")
    fp.close()

    games_elos, games_moves = extract_game_details(lines)

    state_actions = pd.DataFrame()

    games_processed = 0
    for elos, turns in zip(games_elos, games_moves):
        game = Game(elos[0], elos[1])
        for i, turn in enumerate(turns):
            for move in turn:
                # one-hot ?
                # action = np.zeros(8)
                # action[move] = 1

                # or num?
                action = int(move)
                state = game.boards[i % 2].board
                game.make_move(action)
                state_action = [np.append(state.flatten(), [action])]
                state_actions = state_actions.append(pd.DataFrame(state_action), ignore_index=True)

        games_processed += 1
        if games_processed % 10 == 1:
            print("Processed {} games".format(games_processed))
        if games_processed > games_to_process:
            break

    state_actions.to_csv('data/state_action.csv', index=False)


if __name__ == '__main__':
    generate_state_actions(25)
