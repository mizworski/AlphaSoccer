import tensorflow as tf
import numpy as np
from src.soccer.board import Board
import os
import time

input_shape = [1, 11, 9, 12]


class Soccer:
    def __init__(self, k_last_models=5, models_dir='models/policy_networks/policy_gradient'):
        self.player_board = None
        self.env_agent_board = None

        checkpoint_state = tf.train.get_checkpoint_state(models_dir)
        all_checkpoints = list(reversed(checkpoint_state.all_model_checkpoint_paths))

        k_last_models = k_last_models if k_last_models <= len(all_checkpoints) else len(all_checkpoints)

        chosen_model = int(k_last_models * np.random.random())
        model_path = all_checkpoints[chosen_model]
        saver = tf.train.import_meta_graph(model_path + '.meta')

        self.sess = tf.Session()
        saver.restore(self.sess, model_path)
        graph = tf.get_default_graph()
        self.inputs = graph.get_tensor_by_name("shuffle_batch:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.output = graph.get_tensor_by_name("PolicyNetwork/output/BiasAdd:0")

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
            keep_prob = self.keep_prob

            feed_dict = {
                inputs: self.env_agent_board.board.reshape(input_shape),
                keep_prob: 1.0
            }

            env_logits = self.sess.run([self.output], feed_dict=feed_dict)
            env_action = np.argmax(env_logits)

            env_reward, env_turn = self.env_agent_board.make_move(env_action)
            _ = self.player_board.make_move((env_action + 4) % 8)
            reward_after_env_move = -env_reward

            # todo experiment with this shit
            while env_reward == -1:
                ball_pos = self.env_agent_board.get_pos()
                # if there is still move available
                if (self.env_agent_board.board[ball_pos][:8] == np.array([1] * 8)).all():
                    return self.player_board.board.reshape(input_shape), 1, True # reward=1 or reward=0?

                rand_move = int(np.random.rand() * 8)
                env_reward, env_turn = self.env_agent_board.make_move(rand_move)
                _ = self.player_board.make_move((rand_move + 4) % 8)
                reward_after_env_move = -env_reward


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
    env = Soccer(k_last_models=4)

    while True:
        state = env.reset(0, verbose=True)

        for _ in range(100):
            action = input()
            action = int(action)

            state, reward, done = env.step(action, verbose=0)

            if done:
                print('Game ended, reward = {}'.format(reward))
                break
