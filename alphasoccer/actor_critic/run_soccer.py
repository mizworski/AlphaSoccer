import os
import argparse
from alphasoccer.actor_critic.trainer import learn


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--job-dir', type=str,
                        default=None,
                        help='Not being used now. I have to catch it while training on google cloud')
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join('models', 'actor_critic'),
                        help='Directory where model will be saved.')
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join('models', 'logs'),
                        help='Directory where training will be logged.')
    parser.add_argument('--replay_dir', type=str,
                        default=os.path.join('data', 'replays'),
                        help='Directory where replays will be stored.')

    parser.add_argument('--n_total_timesteps', type=int,
                        default=500,
                        help='Number of self play epochs.')
    parser.add_argument('--n_training_steps', type=int,
                        default=1024,
                        help='Number of training steps before evaluation.')
    parser.add_argument('--n_self_play_games', type=int,
                        default=256,
                        help='Number of self play games between trainings.')
    parser.add_argument('--n_evaluation_games', type=int,
                        default=100,
                        help='Number of evaluation games.')
    parser.add_argument('--n_evaluations', type=int,
                        default=10,
                        help='Number of evaluations before starting new self play process.')
    parser.add_argument('--new_best_model_threshold', type=float,
                        default=0.55,
                        help='Win ratio required to set new best player.')

    parser.add_argument('--n_games_in_replay_checkpoint', type=int,
                        default=128,
                        help='Number of games per single pickled history.')
    parser.add_argument('--n_replays', type=int,
                        default=1024,
                        help='Games stored in replay history.')

    parser.add_argument('--n_rollouts', type=int,
                        default=100,
                        help='Number of rollouts per move.')
    parser.add_argument('--training_temperature', type=float,
                        default=1.0,
                        help='Training initial temperature.')
    parser.add_argument('--evaluation_temperature', type=float,
                        default=1.0,
                        help='Evaluation initial temperature.')
    parser.add_argument('--temperature_decay_factor', type=float,
                        default=0.95,
                        help='Decay factor of temperature.')
    parser.add_argument('--moves_before_dacaying', type=int,
                        default=10,
                        help='Number of moves before temperature starts to drop.')

    parser.add_argument('--batch_size', type=int,
                        default=512,
                        help='Size of batch.')
    parser.add_argument('--vf_coef', type=float,
                        default=1.0,
                        help='Value function loss coefficient in total loss.')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-2,
                        help='Initial learning rate.')
    parser.add_argument('--c_puct', type=float,
                        default=1.0,
                        help='PUCT constant.')
    parser.add_argument('--lrschedule', type=str,
                        default='stairs',
                        help='Schedule for learning rate (constant, linear, stairs).')

    parser.add_argument('--n_kernels', type=int,
                        default=128,
                        help='Kernels per conv layer.')
    parser.add_argument('--residual_blocks', type=int,
                        default=10,
                        help='Number of residual blocks in network.')
    parser.add_argument('--reg_factor', type=float,
                        default=1e-3,
                        help='Regularization parameter.')

    parser.add_argument('--skip_first_self_play',
                        default=False,
                        action='store_true',
                        help='Skip first self play phase. Start with training.')
    parser.add_argument('--double_first_self_play',
                        default=False,
                        action='store_true',
                        help='Play twice as many games in first epoch.')

    parser.add_argument('--verbose', type=int,
                        default=1,
                        help='Verbosity level.')

    args = parser.parse_args()

    learn(batch_size=args.batch_size, n_self_play_games=args.n_self_play_games, n_replays=args.n_replays,
          n_total_timesteps=args.n_total_timesteps, initial_temperature=args.training_temperature, vf_coef=args.vf_coef,
          initial_lr=args.learning_rate, evaluation_temperature=args.evaluation_temperature,
          n_training_steps=args.n_training_steps, n_evaluation_games=args.n_evaluation_games,
          n_evaluations=args.n_evaluations, model_dir=args.model_dir,
          new_best_model_threshold=args.new_best_model_threshold, n_rollouts=args.n_rollouts, c_puct=args.c_puct,
          temperature_decay_factor=args.temperature_decay_factor, moves_before_dacaying=args.moves_before_dacaying,
          lrschedule=args.lrschedule, replay_checkpoint_dir=args.replay_dir, log_dir=args.log_dir,
          n_games_in_replay_checkpoint=args.n_games_in_replay_checkpoint,
          skip_first_self_play=args.skip_first_self_play, double_first_self_play=args.double_first_self_play,
          n_kernels=args.n_kernels, reg_fact=args.reg_factor, residual_blocks=args.residual_blocks,
          verbose=args.verbose)


if __name__ == '__main__':
    main()
