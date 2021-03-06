import os

import sys
from tqdm import tqdm

from alphasoccer.actor_critic.model import Model
from alphasoccer.actor_critic.self_play import Runner
from alphasoccer.environment.PaperSoccer import Soccer


def learn(batch_size=1024, n_self_play_games=int(4e3), n_replays=int(3e6), n_total_timesteps=int(1e3),
          initial_temperature=1, vf_coef=1, initial_lr=1e-10, evaluation_temperature=0.5, n_training_steps=16,
          n_evaluation_games=400, n_evaluations=8, model_dir=None, new_best_model_threshold=0.55, n_rollouts=1600,
          c_puct=1, temperature_decay_factor=0.95, moves_before_decaying=8, lrschedule='constant',
          replay_checkpoint_dir=os.path.join('data', 'replays'), log_dir=os.path.join('models', 'logs'),
          n_games_in_replay_checkpoint=200,
          skip_first_self_play=False, n_kernels=128, reg_fact=1e-3,
          residual_blocks=8, verbose=1):
  n_training_timesteps = n_evaluations * n_training_steps

  model = Model(Soccer.observation_space, Soccer.action_space, batch_size=batch_size, vf_coef=vf_coef, lr=initial_lr,
                training_timesteps=n_training_timesteps, lrschedule=lrschedule, model_dir=model_dir, log_dir=log_dir,
                n_kernels=n_kernels, reg_fact=reg_fact, residual_blocks=residual_blocks)
  runner = Runner(model, n_replays=n_replays, c_puct=c_puct, replay_checkpoint_dir=replay_checkpoint_dir,
                  n_games_in_replay_checkpoint=n_games_in_replay_checkpoint, verbose=verbose)

  model_iterations = model.initial_checkpoint_number
  train_iter = 0
  eval_iter = 0
  for epoch in range(n_total_timesteps):
    print("Training epoch = {}".format(epoch), file=sys.stderr)
    if (epoch != 0 or not skip_first_self_play) and n_self_play_games > 0:
      playing_turns = n_self_play_games // n_games_in_replay_checkpoint
      for _ in range(playing_turns):
        runner.run(n_games=n_games_in_replay_checkpoint, initial_temperature=initial_temperature, n_rollouts=n_rollouts,
                 temperature_decay_factor=temperature_decay_factor, moves_before_decaying=moves_before_decaying)

    model.lr.reset_steps()
    for evaluation in range(n_evaluations):
      progress_bar = tqdm(total=n_training_steps)
      for train in range(n_training_steps):
        states, actions, rewards = runner.replay_memory.sample(batch_size)
        model.train(states, actions, rewards, train_iter=train_iter)
        train_iter += 1
        progress_bar.update(1)

      progress_bar.close()
      if n_evaluation_games > 0:
        new_model_won = runner.evaluate(model, n_games=n_evaluation_games,
                                        initial_temperature=evaluation_temperature,
                                        n_rollouts=n_rollouts, new_best_model_threshold=new_best_model_threshold,
                                        temperature_decay_factor=temperature_decay_factor,
                                        moves_before_decaying=moves_before_decaying, eval_iter=eval_iter,
                                        verbose=verbose)
        eval_iter += 1
        if new_model_won:
          model_iterations += 1
          model.update_best_player()
          model.save(model_iterations)
          print('New best player saved iter={}.'.format(model_iterations), file=sys.stderr)
          break

  model.summary_writer.close()
