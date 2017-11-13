import os
from src2.actor_critic.trainer import learn


def main():
    batch_size = 2048
    n_games = int(1e3)
    n_replays = int(2e5)
    n_total_timesteps = int(1e3)
    initial_temperature = 8
    initial_lr = 1e-10
    evaluation_temperature = 0.5
    n_training_steps = 4
    n_evaluations = 4
    verbose = 1

    model_dir = os.path.join('models', 'actor_critic')

    learn(batch_size=batch_size, n_games=n_games, n_replays=n_replays, n_total_timesteps=n_total_timesteps,
          initial_temperature=initial_temperature, initial_lr=initial_lr,
          evaluation_temperature=evaluation_temperature, n_training_steps=n_training_steps, n_evaluations=n_evaluations,
          model_dir=model_dir, verbose=verbose)


if __name__ == '__main__':
    main()
