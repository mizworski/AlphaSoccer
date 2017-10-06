import tensorflow as tf
import numpy as np
from src.soccer.board import Board
import os
import time

input_shape = [1, 11, 9, 12]


class Soccer:
    def __init__(self, k_last_models=5, models_dir='models/tf'):
        self.player_board = None
        self.env_agent_board = None

        models_dirs = os.listdir(models_dir)
        chosen_model = int(np.random.random(k_last_models))
        model_dir = models_dirs[chosen_model]

        model_path = tf.train.latest_checkpoint(model_dir)
        saver = tf.train.import_meta_graph(model_path + '.meta')

        self.sess = tf.Session()
        saver.restore(self.sess, model_path)
        graph = tf.get_default_graph()
        self.inputs = graph.get_tensor_by_name("shuffle_batch:0")
        self.output = graph.get_tensor_by_name("SLNet/output/BiasAdd:0")

    def step(self, action, verbose=0):
        reward_after_player_move, bonus_move = self.player_board.make_move(action)
        reward_after_env_move = reward_after_player_move
        if reward_after_player_move != 0:
            return self.player_board.board.reshape(input_shape), reward_after_player_move, True

        self.env_agent_board.make_move((action + 4) % 8)

        if verbose:
            # time.sleep(0.1)
            # os.system('clear')
            self.player_board.print_board()
        if bonus_move:
            return self.player_board.board.reshape(input_shape), reward_after_player_move, False

        env_turn = True

        while env_turn:
            inputs = self.inputs
            feed_dict = {
                inputs: self.env_agent_board.board.reshape(input_shape)
            }

            env_logits = self.sess.run([self.output], feed_dict=feed_dict)
            env_action = np.argmax(env_logits)

            env_reward, env_turn = self.env_agent_board.make_move(env_action)

            reward_after_env_move, _ = self.player_board.make_move((env_action + 4) % 8)

            print('env rew = {}'.format(env_reward))
            print('my rew = {}'.format(reward_after_env_move))
            while env_reward == -1:
                rand_move = int(np.random.rand() * 7.99)
                env_reward, env_turn = self.env_agent_board.make_move(rand_move)
                reward_after_env_move, _ = self.player_board.make_move((rand_move + 4) % 8)

            print(env_action)

            if verbose:
                self.player_board.print_board()

        return self.player_board.board.reshape(input_shape), reward_after_env_move, reward_after_env_move != 0

    def reset(self, starting_game=True, verbose=0):
        self.player_board = Board()
        self.env_agent_board = Board()

        if not starting_game:
            inputs = self.inputs
            feed_dict = {
                inputs: self.env_agent_board.board.reshape(input_shape)
            }

            env_logits = self.sess.run([self.output], feed_dict=feed_dict)
            env_action = np.argmax(env_logits)

            self.env_agent_board.make_move(env_action)
            self.player_board.make_move((env_action + 4) % 8)

            if verbose:
                self.player_board.print_board()

        return self.player_board.board.reshape(input_shape)


if __name__ == '__main__':
    env = Soccer()

    while True:
        state = env.reset(np.random.rand() < 0.5, verbose=True)

        for _ in range(100):
            action = input()
            action = int(action)

            state, reward, done = env.step(action, verbose=1)

            if done:
                print('Game ended, reward = {}'.format(reward))
                break
