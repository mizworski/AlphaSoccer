import os
from src.actor_critic.trainer import learn


def main():
    batch_size = 1024
    n_self_play_games = int(100)
    n_replays = int(1e5)
    initial_temperature = 1
    vf_coef = 1
    initial_lr = 1e-3
    evaluation_temperature = 1
    n_training_steps = 512
    n_evaluations = 4
    verbose = 1
    new_best_model_threshold = 0.55
    c_puct = 1
    n_evaluation_games = 40
    n_rollouts = 400

    # todo better name
    n_total_timesteps = 500

    temperature_decay_factor = 0.95
    moves_before_dacaying = 8
    lrschedule = 'constant'

    model_dir = os.path.join('models', 'actor_critic')

    learn(batch_size=batch_size, n_self_play_games=n_self_play_games, n_replays=n_replays,
          n_total_timesteps=n_total_timesteps, initial_temperature=initial_temperature, vf_coef=vf_coef,
          initial_lr=initial_lr, evaluation_temperature=evaluation_temperature, n_training_steps=n_training_steps,
          n_evaluation_games=n_evaluation_games, n_evaluations=n_evaluations, model_dir=model_dir,
          new_best_model_threshold=new_best_model_threshold, n_rollouts=n_rollouts, c_puct=c_puct,
          temperature_decay_factor=temperature_decay_factor, moves_before_dacaying=moves_before_dacaying,
          lrschedule=lrschedule, verbose=verbose)


if __name__ == '__main__':
    main()
