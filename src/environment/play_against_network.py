import os
import numpy as np

from src.actor_critic.mcts import MCTS
from src.actor_critic.model import Model
from src.environment.PaperSoccer import Soccer


def play(batch_size=2048, n_total_timesteps=int(1e3),
         initial_temperature=1, initial_lr=1e-10, n_training_steps=16,
         model_dir=None, n_rollouts=1600,
         c_puct=1, verbose=1):
    n_training_timesteps = n_total_timesteps * n_training_steps
    model = Model(Soccer.observation_space, Soccer.action_space, batch_size=batch_size, lr=initial_lr,
                  training_timesteps=n_training_timesteps, model_dir=model_dir)
    envs = [Soccer(), Soccer()]

    mcts = MCTS(envs, model, n_rollouts=n_rollouts, c_puct=c_puct)

    while True:
        winner = play_single_game(envs, mcts, initial_temperature=initial_temperature,
                                  starting_player=np.random.choice(2))
        if winner == 1:
            print("Player won")
        else:
            print("Player lost")

def play_single_game(envs, mcts, starting_player=0, initial_temperature=1.0,
                     temperature_decay_factor=0.95):
    for i in range(2):
        # is player0 starts then env1 has to set starting player as player1 (cause real player1 is not starting)
        starting_from_i_perspective = abs(i - starting_player)
        envs[i].reset(starting_game=starting_from_i_perspective)

    mcts.reset(starting_player=starting_player)

    done = False
    player_turn = None
    reward = None
    temperature = initial_temperature

    moves = 0
    while not done:
        player_turn = envs[0].get_player_turn()
        envs[1].print_board()
        if player_turn == 0:

            action, pi = mcts.select_action(player_turn, temperature=temperature)
        else:
            action = input('Your turn ')
            action = int(action)

        action_opposite_player_perspective = (action + 4) % 8
        _, reward, done = envs[player_turn].step(action)
        _ = envs[1 - player_turn].step(action_opposite_player_perspective)

        mcts.step(action)

        moves += 1
        if moves > 32 and temperature > 5e-2:
            temperature *= temperature_decay_factor

    if (player_turn == 0 and reward > 0) or (player_turn == 1 and reward < 0):
        winner = 0
    else:
        winner = 1


    return winner

if __name__ == '__main__':
    model_dir = os.path.join('models', 'actor_critic')
    n_rollouts = 1600
    play(n_rollouts=n_rollouts, model_dir=model_dir)
