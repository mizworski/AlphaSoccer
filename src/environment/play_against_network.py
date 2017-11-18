import os
import random

from src.actor_critic.mcts import MCTS
from src.actor_critic.model import Model
from src.environment.PaperSoccer import Soccer


def main():
    envs = [Soccer(), Soccer()]
    model_dir = os.path.join('models', 'actor_critic')
    temperature = 0.5

    model = Model(Soccer.observation_space, Soccer.action_space, batch_size=None, lr=None,
                  training_timesteps=None, model_dir=model_dir, verbose=None)


    while True:
        for i in range(2):
            envs[i].reset(i)

        player = random.choice((0, 1))
        print(player)
        done = False
        mcts = MCTS(envs, model, temperature=temperature)
        while not done:
            player_turn = envs[0].get_player_turn()
            envs[player].print_board()
            if player_turn == player:
                action = input('Your turn ')
                action = int(action)
            else:
                _ = envs[player_turn].board.state
                action, _ = mcts.select_action(player_turn)

            _, reward, done = envs[player_turn].step(action)
            action_opposite_player_perspective = (action + 4) % 8
            _ = envs[1 - player_turn].step(action_opposite_player_perspective)

        if (player_turn == player and reward > 0) or (player_turn != player and reward < 0):
            print("You won")
        else:
            print("You lost")


if __name__ == '__main__':
    main()
