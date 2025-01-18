import gymnasium as gym
import gym_sde_is

from rl_for_gym.dpg.reinforce_deterministic_core import ReinforceDeterministic
from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.plots import *


def main():
    parser = get_base_parser()
    parser.description = 'Run model-based deterministic policy gradient for the sde \
                          importance sampling environment with a ol toy example.'
    parser.add_argument(
        '--d',
        type=int,
        default=1,
        help='Dimension of the environment. Default: 1',
    )
    parser.add_argument(
        '--alpha-i',
        type=float,
        default=1.,
        help='Set i-th component of the barrier height parameter of the given potential. Default: 1.',
    )
    parser.add_argument(
        '--alpha-j',
        type=float,
        default=1.,
        help='Set j-th component of the barrier height parameter of the given potential. Default: 1.',
    )
    parser.add_argument(
        '--alpha-k',
        type=float,
        default=1.,
        help='Set k-th component of the barrier height parameter of the given potential. Default: 1.',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1.,
        help='Set inverse of the temperature. Default: 1.',
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

    # reinforce deterministic agent
    breakpoint()
    agent = ReinforceDeterministic(
        env, env.unwrapped.__str__(), env._max_episode_steps, args.expectation_type, args.return_type, args.gamma,
        args.n_layers, args.d_hidden, args.batch_size, args.lr, args.n_grad_iterations, args.seed,
        args.estimate_z, args.batch_size_z, args.mini_batch_size, args.mini_batch_size_type,
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
