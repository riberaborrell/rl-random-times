import gymnasium as gym
import numpy as np

from rl_random_times.spg.stochastic_pg_core import ReinforceStochastic
from rl_random_times.wrappers.episodic_reacher import EpisodicReacherEnv
from rl_random_times.utils.base_parser import get_base_parser
from rl_random_times.utils.path import get_reacher_env_str
from rl_random_times.utils.plots import plot_y_per_grad_iteration

def make_env(env_id, threshold_dist, threshold_vel, reward_ctrl_weight):
    def _init():
        env = gym.make(env_id)
        env = EpisodicReacherEnv(env, threshold_dist, threshold_vel,
                                 reward_ctrl_weight=reward_ctrl_weight)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=int(1e6))
        return env
    return _init

def main():
    parser = get_base_parser()
    parser.add_argument(
        '--threshold-dist',
        type=float,
        default=0.05,
        help='Threshold distance for Episodic Reacher environment',
    )
    parser.add_argument(
        '--threshold-vel',
        type=float,
        default=1.,
        help='Threshold angular velocity for Episodic Reacher environment',
    )
    parser.add_argument(
        '--reward-ctrl-weight',
        type=float,
        default=0.1,
        help='Reward control weight parameter of the Reacher environment.',
    )
    args = parser.parse_args()

    assert 'Reacher' in args.env_id, 'This script only works with the Reacher environment.'

    # create vectorized environment
    env = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.threshold_dist, args.threshold_vel, args.reward_ctrl_weight) for _ in range(args.batch_size)]
    )

    # environment name
    env_name = get_reacher_env_str(
        args.env_id, args.threshold_dist, args.threshold_vel, args.reward_ctrl_weight,
    )

    # reinforce stochastic agent
    agent = ReinforceStochastic(
        env, env_name, env.envs[0]._max_episode_steps, args.expectation_type, args.return_type, args.gamma,
        args.n_layers, args.d_hidden, args.batch_size, args.lr, args.n_grad_iterations, args.seed,
        args.gaussian_policy_type, args.policy_noise, args.estimate_z,
        args.batch_size_z, args.mini_batch_size, args.mini_batch_size_type,
        args.optim_type, args.scheduled_lr, args.lr_final,
    )

    # run
    succ, data = agent.run_reinforce(
        log_freq=args.log_freq,
        backup_freq=args.backup_freq,
        live_plot_freq=args.live_plot_freq,
        load=args.load,
    )
    env.close()

    # do plots
    if not args.plot or not succ:
        return

    # plot returns and time steps
    x = np.arange(args.n_grad_iterations + 1)
    plot_y_per_grad_iteration(x, data['mean_returns'], title='Mean return', run_window=10, legend=True)
    plot_y_per_grad_iteration(x, data['mean_lengths'], title='Mean time steps', run_window=10)
    plot_y_per_grad_iteration(x, data['losses'], title='Losses', run_window=100)


if __name__ == '__main__':
    main()
