import os

from src.actor_critic.model import ParallelModel as Model
from src.actor_critic.self_play import Runner
from src.environment.PaperSoccer import Soccer


def learn(batch_size=1024, n_self_play_games=int(4e3), n_replays=int(3e6), n_total_timesteps=int(1e3),
          initial_temperature=1, vf_coef=1, initial_lr=1e-10, evaluation_temperature=0.5, n_training_steps=16,
          n_evaluation_games=400, n_evaluations=8, model_dir=None, new_best_model_threshold=0.55, n_rollouts=1600,
          c_puct=1, temperature_decay_factor=0.95, moves_before_dacaying=8, lrschedule='constant',
          replay_checkpoint_dir=os.path.join('data', 'replays'), checkpoint_every_n_transitions=200,
          skip_first_self_play=False, verbose=1):
    n_training_timesteps = n_total_timesteps * n_training_steps
    log_every_n_train_steps = max(2, n_training_steps // 16)

    model = Model(Soccer.observation_space, Soccer.action_space, batch_size=batch_size, vf_coef=vf_coef, lr=initial_lr,
                  training_timesteps=n_training_timesteps, lrschedule=lrschedule, model_dir=model_dir)
    runner = Runner(model, n_replays=n_replays, c_puct=c_puct, replay_checkpoint_dir=replay_checkpoint_dir,
                    checkpoint_every_n_transitions=checkpoint_every_n_transitions, verbose=verbose)

    model_iterations = model.initial_checkpoint_number

    for epoch in range(n_total_timesteps):
        if epoch != 0 or not skip_first_self_play:
            runner.run(n_games=n_self_play_games, initial_temperature=initial_temperature, n_rollouts=n_rollouts,
                       temperature_decay_factor=temperature_decay_factor, moves_before_dacaying=moves_before_dacaying)

        for _ in range(n_evaluations):
            for train in range(n_training_steps):
                states, actions, rewards = runner.replay_memory.sample(batch_size)
                policy_loss, value_loss = model.train(states, actions, rewards)

                if n_training_steps == 1 or train % log_every_n_train_steps == log_every_n_train_steps - 1:
                    print("Training step {}".format(train))
                    print("policy_loss", float(policy_loss))
                    print("value_loss", float(value_loss))

            new_model_won = runner.evaluate(model, n_games=n_evaluation_games,
                                            initial_temperature=evaluation_temperature,
                                            n_rollouts=n_rollouts, new_best_model_threshold=new_best_model_threshold,
                                            temperature_decay_factor=temperature_decay_factor,
                                            moves_before_dacaying=moves_before_dacaying,
                                            verbose=verbose)

            if new_model_won:
                model_iterations += 1
                model.update_best_player()
                model.save(model_iterations)
                print('New best player saved iter = {}.'.format(model_iterations))
                break
