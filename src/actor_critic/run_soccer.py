import os
from src.actor_critic.trainer import learn


def main():
    n_total_timesteps = 500
    batch_size = 512
    n_training_steps = 1024
    n_rollouts = 100

    vf_coef = 1
    initial_lr = 1e-3
    lrschedule = 'constant'
    c_puct = 1

    initial_temperature = 1
    evaluation_temperature = 1
    temperature_decay_factor = 0.95
    moves_before_dacaying = 10

    n_evaluation_games = 100
    n_evaluations = 5
    new_best_model_threshold = 0.55

    n_self_play_games = int(250)
    checkpoint_every_n_transitions = 200
    n_replays = int(5e4)

    skip_first_self_play = False

    verbose = 1
    model_dir = os.path.join('models', 'actor_critic')
    replay_checkpoint_dir = os.path.join('data', 'replays')

    learn(batch_size=batch_size, n_self_play_games=n_self_play_games, n_replays=n_replays,
          n_total_timesteps=n_total_timesteps, initial_temperature=initial_temperature, vf_coef=vf_coef,
          initial_lr=initial_lr, evaluation_temperature=evaluation_temperature, n_training_steps=n_training_steps,
          n_evaluation_games=n_evaluation_games, n_evaluations=n_evaluations, model_dir=model_dir,
          new_best_model_threshold=new_best_model_threshold, n_rollouts=n_rollouts, c_puct=c_puct,
          temperature_decay_factor=temperature_decay_factor, moves_before_dacaying=moves_before_dacaying,
          lrschedule=lrschedule, replay_checkpoint_dir=replay_checkpoint_dir,
          checkpoint_every_n_transitions=checkpoint_every_n_transitions, skip_first_self_play=skip_first_self_play,
          verbose=verbose)


if __name__ == '__main__':
    main()
