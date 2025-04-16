import gymnasium as gym
import numpy as np

from rl_random_times.spg.stochastic_pg_core import ReinforceStochastic
from rl_random_times.wrappers.episodic_swimmer import EpisodicSwimmerEnv
from rl_random_times.utils.base_parser import get_base_parser
from rl_random_times.utils.path import get_swimmer_env_str
from rl_random_times.utils.plots import plot_y_per_grad_iteration

def make_env(env_id, threshold_fwd_dist):
    def _init():
        env = gym.make(env_id)
        env = EpisodicSwimmerEnv(env, threshold_fwd_dist)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=int(1e6))
        return env
    return _init

def main():
    parser = get_base_parser()
    parser.add_argument(
        '--threshold-fwd-dist',
        type=float,
        default=1.,
        help='Set forward distance goal in the x-direction',
    )
    args = parser.parse_args()

    # batch size
    K = args.batch_size if args.expectation_type == 'random-time' else args.batch_size_z

    # create vectorized environment
    assert 'Swimmer' in args.env_id, 'This script only works with the Swimmer environment.'
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.threshold_fwd_dist) for _ in range(K)]
    )

    # environment name
    env_name = get_swimmer_env_str(args.env_id, args.threshold_fwd_dist)

    # reinforce stochastic agent
    agent = ReinforceStochastic(
        envs,
        env_name=env_name,
        n_steps_lim=envs.envs[0]._max_episode_steps,
        expectation_type=args.expectation_type,
        return_type=args.return_type,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden_layer,
        batch_size=args.batch_size,
        lr=args.lr,
        n_grad_iterations=args.n_grad_iterations,
        seed=args.seed,
        policy_type=args.gaussian_policy_type,
        policy_noise=args.policy_noise,
        estimate_z=args.estimate_z,
        batch_size_z=args.batch_size_z,
        mini_batch_size=args.mini_batch_size,
        mini_batch_size_type=args.mini_batch_size_type,
        optim_type=args.optim_type,
        scheduled_lr=args.scheduled_lr,
        lr_final=args.lr_final,
        norm_returns=args.norm_returns,
    )

    # run reinforce with random time horizon 
    succ, data = agent.run_reinforce(
        log_freq=args.log_freq,
        backup_freq=args.backup_freq,
        load=args.load,
    )
    envs.close()

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
