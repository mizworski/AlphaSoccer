from src.actor_critic.model import Model
from src.actor_critic.self_play import Runner
from src.environment.PaperSoccer import Soccer


def learn(batch_size=2048, n_self_play_games=int(4e3), n_replays=int(3e5), n_total_timesteps=int(1e3),
          initial_temperature=1, vf_coef=1, initial_lr=1e-10, evaluation_temperature=0.5, n_training_steps=16,
          n_evaluation_games=400, n_evaluations=8, model_dir=None, new_best_model_threshold=0.55, n_rollouts=1600,
          c_puct=1, temperature_decay_factor=0.95, moves_before_dacaying=8, verbose=1):
    n_training_timesteps = n_total_timesteps * n_training_steps
    log_every_n_train_steps = max(2, n_training_steps // 16)

    model = Model(Soccer.observation_space, Soccer.action_space, batch_size=batch_size, vf_coef=vf_coef, lr=initial_lr,
                  training_timesteps=n_training_timesteps, model_dir=model_dir, verbose=verbose)
    runner = Runner(model, n_replays=n_replays, c_puct=c_puct)

    model_iterations = 0
    for epoch in range(n_total_timesteps):
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

            new_best_player = runner.evaluate(model, n_games=n_evaluation_games,
                                              initial_temperature=evaluation_temperature,
                                              n_rollouts=n_rollouts, new_best_model_threshold=new_best_model_threshold,
                                              temperature_decay_factor=temperature_decay_factor,
                                              moves_before_dacaying=moves_before_dacaying,
                                              verbose=verbose)

            if new_best_player:
                model_iterations += 1
                model.save(model_iterations)
                model.update_best_player()
                print('New best player saved.')
                break
