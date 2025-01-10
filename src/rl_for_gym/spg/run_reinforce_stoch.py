import envpool
import gymnasium as gym
import numpy as np
import torch

from rl_for_gym.spg.reinforce_stochastic_core import ReinforceStochastic
from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.evaluate import simulate_learnt_policy_episode
from rl_for_gym.utils.plots import plot_y_per_grad_iteration

def main():
    args = get_base_parser().parse_args()

    # batch size
    K = args.batch_size if args.expectation_type == 'random-time' else args.batch_size_z

    # create gym env 
    if not args.envpool:
        env = gym.make(args.env_id, max_episode_steps=args.n_steps_lim, is_vectorized=True)
    elif args.n_steps_lim is None:
        env = envpool.make_gymnasium(args.env_id, num_envs=K, seed=args.seed)
    else:
        env = envpool.make_gymnasium(args.env_id, num_envs=K,
                                     seed=args.seed, max_episode_steps=args.n_steps_lim)

    # reinforce stochastic agent
    agent = ReinforceStochastic(
        env, args.env_id, args.n_steps_lim, args.expectation_type, args.return_type, args.gamma,
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
    if not succ:
        return

    # render agent
    if args.render:
        env = gym.make(args.env_id, max_episode_steps=args.n_steps_lim, render_mode='human')
        agent.load_backup_model(data, i=args.n_grad_iterations)
        simulate_learnt_policy_episode(env, agent.policy, args)

    # do plots
    if not args.plot:
        return

    # plot returns and time steps
    x = np.arange(args.n_grad_iterations + 1)
    plot_y_per_grad_iteration(x, data['mean_returns'], title='Mean return', run_window=10, legend=True)
    plot_y_per_grad_iteration(x, data['mean_lengths'], title='Mean time steps', run_window=10)
    plot_y_per_grad_iteration(x, data['losses'], title='Losses', run_window=100)
    if args.scheduled_lr:
        plot_y_per_grad_iteration(x, data['lrs'], title='Learning rates', plot_scale='semilogy', run_window=1)

if __name__ == '__main__':
    main()
