import time
import tensorflow as tf

from src2.actor_critic.self_play import Runner
from src2.actor_critic.model import Model
from src2.actor_critic.utils import Scheduler, explained_variance
from src2.environment.PaperSoccer import Soccer


def learn():
    batch_size = 1024
    n_games = int(1e3)
    n_replays = int(2e4)
    n_total_timesteps = int(1e3)
    initial_temperature = 16
    initial_lr = 1e-7
    evaluation_temperature = 0.25
    n_training_steps = 128
    n_evaluations = 4
    n_training_timesteps = n_total_timesteps * n_training_steps
    log_every_n_train_steps = 64

    model = Model(Soccer.observation_space, Soccer.action_space, batch_size=batch_size, lr=initial_lr,
                  training_timesteps=n_training_timesteps)
    runner = Runner(model, n_replays=n_replays)
    temperature = Scheduler(initial_temperature, n_total_timesteps, 'linear')

    for epoch in range(n_total_timesteps):
        runner.run(n_games=n_games, temperature=temperature.value())
        for _ in range(n_evaluations):
            for train in range(n_training_steps):
                states, actions, rewards, values = runner.replay_memory.sample(batch_size)
                policy_loss, value_loss, policy_entropy = model.train(states, rewards, actions, values)

                if train % log_every_n_train_steps == log_every_n_train_steps - 1:
                    ev = explained_variance(values, rewards)
                    print("policy_entropy", float(policy_entropy))
                    print("value_loss", float(value_loss))
                    print("explained_variance", float(ev))

            new_best_player = runner.evaluate(model, temperature=evaluation_temperature, verbose=2)

            if new_best_player:
                break


def main():
    learn()


if __name__ == '__main__':
    main()
