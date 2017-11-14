from src2.actor_critic.self_play import Runner
from src2.actor_critic.model import Model
from src2.actor_critic.utils import Scheduler, explained_variance
from src2.environment.PaperSoccer import Soccer


def learn(batch_size=2048, n_games=int(4e3), n_replays=int(3e5), n_total_timesteps=int(1e3), initial_temperature=8,
          initial_lr=1e-10, evaluation_temperature=0.5, n_training_steps=16, n_evaluations=8, model_dir=None,
          new_best_model_threshold=0.55, verbose=1):
    n_training_timesteps = n_total_timesteps * n_training_steps
    log_every_n_train_steps = n_training_steps // 2

    model = Model(Soccer.observation_space, Soccer.action_space, batch_size=batch_size, lr=initial_lr,
                  training_timesteps=n_training_timesteps, model_dir=model_dir, verbose=verbose)
    runner = Runner(model, n_replays=n_replays)
    temperature = Scheduler(initial_temperature, n_total_timesteps, 'linear')

    model_iterations = 0
    for epoch in range(n_total_timesteps):
        runner.run(n_games=n_games, temperature=temperature.value())
        for _ in range(n_evaluations):
            for train in range(n_training_steps):
                states, actions, rewards, values = runner.replay_memory.sample(batch_size)
                policy_loss, value_loss, policy_entropy = model.train(states, actions, rewards, values)

                if train % log_every_n_train_steps == log_every_n_train_steps - 1:
                    ev = explained_variance(values, rewards)
                    print("policy_entropy", float(policy_entropy))
                    print("value_loss", float(value_loss))
                    print("explained_variance", float(ev))

            new_best_player = runner.evaluate(model, temperature=evaluation_temperature,
                                              new_best_model_threshold=new_best_model_threshold, verbose=verbose)

            if new_best_player:
                model_iterations += 1
                model.save(model_iterations)
                model.update_best_player()
                print('model saved')
                break
