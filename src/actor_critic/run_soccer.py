import os
from src.actor_critic.trainer import learn


def main():
    batch_size = 16
    n_games = int(4e1)
    n_replays = int(2e5)
    # todo better name
    n_total_timesteps = int(1e3)
    initial_temperature = 8
    initial_lr = 1e-8
    evaluation_temperature = 0.1
    n_training_steps = 2
    n_evaluations = 32
    verbose = 1
    new_best_model_threshold = 0.55
    n_rollouts = 16

    model_dir = os.path.join('models', 'actor_critic')

    learn(batch_size=batch_size, n_games=n_games, n_replays=n_replays, n_total_timesteps=n_total_timesteps,
          initial_temperature=initial_temperature, initial_lr=initial_lr, evaluation_temperature=evaluation_temperature,
          n_training_steps=n_training_steps, n_evaluations=n_evaluations, model_dir=model_dir,
          new_best_model_threshold=new_best_model_threshold, n_rollouts=n_rollouts, verbose=verbose)


if __name__ == '__main__':
    main()
