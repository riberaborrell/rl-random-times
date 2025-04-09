import envpool
import gymnasium as gym
import numpy as np

from rl_random_times.spg.stochastic_pg_core import ReinforceStochastic
from rl_random_times.utils.base_parser import get_base_parser
from rl_random_times.utils.path import get_hopper_env_str
from rl_random_times.utils.plots import plot_y_per_grad_iteration

def main():
    parser = get_base_parser()
    parser.add_argument(
        '--healthy-reward',
        type=float,
        default=1.,
    )
    parser.add_argument(
        '--forward-reward',
        type=float,
        default=1.,
    )
    parser.add_argument(
        '--ctrl-cost',
        type=float,
        default=1e-3,
    )
    args = parser.parse_args()

    assert 'Hopper' in args.env_id, 'This script only works with the Hopper environment.'

    # environment parameters
    kwargs = {}

    # time horizon
    assert args.n_steps_lim is not None, 'n_steps_lim must be set.'
    kwargs['max_episode_steps'] = args.n_steps_lim

    # batch size
    K = args.batch_size if args.expectation_type == 'random-time' else args.batch_size_z

    # hopper parameters
    kwargs['healthy_reward'] = args.healthy_reward
    kwargs['forward_reward_weight'] = args.forward_reward
    kwargs['ctrl_cost_weight'] = args.ctrl_cost

    # create env 
    if args.env_type == 'gym':
        env = gym.make_vec(args.env_id, num_envs=K, vectorization_mode="sync", **kwargs)
    elif args.env_type == 'envpool':
        env = envpool.make_gymnasium(args.env_id, num_envs=K, seed=args.seed, **kwargs)
    else: # custom vectorized
        raise ValueError(f'Custom vectorized environment for Hopper is not implemented.')

    # environment name
    env_name = get_hopper_env_str(
        args.env_id, args.healthy_reward, args.forward_reward, args.ctrl_cost,
    )

    # reinforce stochastic agent
    agent = ReinforceStochastic(
        env=env,
        env_name=env_name,
        n_steps_lim=args.n_steps_lim,
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

    # run
    succ, data = agent.run_reinforce(
        log_freq=args.log_freq,
        backup_freq=args.backup_freq,
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
