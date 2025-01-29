import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np

from rl_random_times.spg.stochastic_pg_core import ReinforceStochastic
from rl_random_times.wrappers.episodic_swimmer import EpisodicSwimmerEnv
from rl_random_times.utils.base_parser import get_base_parser
from rl_random_times.utils.path import get_swimmer_params_str
from rl_random_times.utils.plots import plot_y_per_grad_iteration

def make_env(env_id, threshold_fwd_dist):
    def _init():
        env = gym.make(env_id)
        env = EpisodicSwimmerEnv(env, threshold_fwd_dist)
        env = TimeLimit(env, max_episode_steps=int(1e6))
        return env
    return _init

def main():
    parser = get_base_parser()
    parser.add_argument(
        '--threshold-fwd-dist',
        type=float,
        default=1.,
        help='Set forward distance goal in the x-direction. Default: 1.',
    )
    args = parser.parse_args()

    # create vectorized environment
    env = gym.vector.SyncVectorEnv(
        [make_env("Swimmer-v5", args.threshold_fwd_dist) for _ in range(args.batch_size)]
    )

    # environment name
    env_name = '{}__{}'.format(
        env.envs[0].spec.id,
        get_swimmer_params_str(args.threshold_fwd_dist)
    )

    # reinforce stochastic agent
    agent = ReinforceStochastic(
        env, env_name, env.envs[0]._max_episode_steps, args.expectation_type, args.return_type, args.gamma,
        args.n_layers, args.d_hidden, args.batch_size, args.lr, args.n_grad_iterations, args.seed,
        args.gaussian_policy_type, args.policy_noise, args.estimate_z,
        args.batch_size_z, args.mini_batch_size, args.mini_batch_size_type,
        args.optim_type, args.scheduled_lr, args.lr_final,
    )

    # run reinforce with random time horizon 
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
