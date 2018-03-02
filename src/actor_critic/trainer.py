from src.actor_critic.self_play import Runner
from src.actor_critic.model import Model
from src.actor_critic.utils import Scheduler, explained_variance
from src.environment.PaperSoccer import Soccer


def learn(batch_size=2048, n_self_play_games=int(4e3), n_replays=int(3e5), n_total_timesteps=int(1e3),
          initial_temperature=8, initial_lr=1e-10, evaluation_temperature=0.5, n_training_steps=16,
          n_evaluation_games=400, n_evaluations=8, model_dir=None, new_best_model_threshold=0.55, n_rollouts=1600,
          c_puct=1, verbose=1):
    n_training_timesteps = n_total_timesteps * n_training_steps
    log_every_n_train_steps = n_training_steps // 2

    model = Model(Soccer.observation_space, Soccer.action_space, batch_size=batch_size, lr=initial_lr,
                  training_timesteps=n_training_timesteps, model_dir=model_dir, verbose=verbose)
    runner = Runner(model, n_replays=n_replays, c_puct=c_puct)
    temperature_scheduler = Scheduler(initial_temperature, n_total_timesteps, 'linear')

    model_iterations = 0
    for epoch in range(n_total_timesteps):
        # todo temperature should be decreasing as we are getting deeper into tree instead decreasing with epochs
        # temperature = temperature_scheduler.value()
        temperature = 1
        runner.run(n_games=n_self_play_games, temperature=temperature, n_rollouts=n_rollouts)
        for _ in range(n_evaluations):
            for train in range(n_training_steps):
                states, actions, rewards = runner.replay_memory.sample(batch_size)
                policy_loss, value_loss, policy_entropy = model.train(states, actions, rewards)

                if n_training_steps == 1 or train % log_every_n_train_steps == log_every_n_train_steps - 1:
                    print("policy_entropy", float(policy_entropy))
                    print("value_loss", float(value_loss))

            new_best_player = runner.evaluate(model, n_games=n_evaluation_games, temperature=evaluation_temperature,
                                              n_rollouts=n_rollouts,    new_best_model_threshold=new_best_model_threshold,
                                              verbose=verbose)

            if new_best_player:
                model_iterations += 1
                model.save(model_iterations)
                model.update_best_player()
                print('New best player saved.')
                break
