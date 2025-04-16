import gymnasium as gym
import gym_sde_is

from rl_random_times.dpg.deterministic_pg_core import ModelBasedDeterministicPG
from rl_random_times.utils.base_parser import get_base_parser
from rl_random_times.utils.plots import *


def main():
    parser = get_base_parser()
    parser.description = 'Run model-based deterministic policy gradient for the sde \
                          importance sampling environment with a ol toy example.'
    parser.add_argument(
        '--d',
        type=int,
        default=1,
        help='the dimension of the environment',
    )
    parser.add_argument(
        '--alpha-i',
        type=float,
        default=1.,
        help='the i-th component of the barrier height parameter of the given potential',
    )
    parser.add_argument(
        '--alpha-j',
        type=float,
        default=1.,
        help='the j-th component of the barrier height parameter of the given potential',
    )
    parser.add_argument(
        '--alpha-k',
        type=float,
        default=1.,
        help='the k-th component of the barrier height parameter of the given potential',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1.,
        help='the inverse of the temperature',
    )
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-doublewell-nd-asym-mgf-v0',
        d=args.d,
        dt=0.01,
        beta=args.beta,
        alpha_i=args.alpha_i,
        alpha_j=args.alpha_j,
        alpha_k=args.alpha_k,
        state_init_dist='delta',
        max_episode_steps=int(1e6),
        is_vectorized=True,
    )

    # model-based deterministic pg agent 
    agent = ModelBasedDeterministicPG(
        env=env,
        env_name=env.unwrapped.__str__(),
        n_steps_lim=env._max_episode_steps,
        gamma=args.gamma,
        expectation_type=args.expectation_type,
        return_type=args.return_type,
        estimate_z=args.estimate_z,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden_layer,
        optim_type=args.optim_type,
        batch_size=args.batch_size,
        batch_size_z=args.batch_size_z,
        mini_batch_size_type=args.mini_batch_size_type,
        mini_batch_size=args.mini_batch_size,
        lr=args.lr,
        n_grad_iterations=args.n_grad_iterations,
        seed=args.seed,
        scheduled_lr=args.scheduled_lr,
        lr_final=args.lr_final,
        norm_returns=args.norm_returns,
        cuda=args.cuda,
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

    # get backup iterations
    iterations = np.arange(0, args.n_grad_iterations + args.backup_freq, args.backup_freq)

    # plot statistics
    x = np.arange(data['n_grad_iterations']+1)
    plot_y_per_grad_iteration(x, data['mean_returns'], title='Mean return', run_window=10, legend=True)
    plot_y_per_grad_iteration(x, data['mean_lengths'], title='Mean time steps', run_window=10)
    plot_y_per_grad_iteration(x, data['losses'], title='Losses', run_window=100)
    if args.scheduled_lr:
        plot_y_per_grad_iteration(x, data['lrs'], title='Learning rates', plot_scale='semilogy', run_window=1)

if __name__ == "__main__":
    main()
