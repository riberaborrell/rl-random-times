import envpool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_for_gym.spg.reinforce_stochastic_core import ReinforceStochastic
from rl_for_gym.spg.models import GaussianPolicyLearntCov
from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.plots import plot_y_per_grad_iteration

def main():
    args = get_base_parser().parse_args()

    # restrict to environments with custom vectorized implementation
    #assert args.env_id in ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0"], ''

    # create gym env 
    #env = gym.make(args.env_id, max_episode_steps=args.n_steps_lim, is_vectorized=True)
    if args.n_steps_lim is None:
        env = envpool.make_gymnasium(args.env_id, num_envs=args.batch_size, seed=args.seed)
    else:
        env = envpool.make_gymnasium(args.env_id, num_envs=args.batch_size,
                                     seed=args.seed, max_episode_steps=args.n_steps_lim)


    # reinforce stochastic agent
    agent = ReinforceStochastic(env, args.env_id, args.expectation_type, args.return_type, args.gamma,
                                args.n_layers, args.d_hidden, args.batch_size, args.lr, args.n_grad_iterations, args.seed,
                                args.gaussian_policy_type, args.policy_noise, args.estimate_z,
                                args.mini_batch_size, args.mini_batch_size_type,
                                args.replay_size, args.optim_type)

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
    plot_y_per_grad_iteration(x, data['mean_returns'], run_window=10, title='Mean return', legend=True)
    plot_y_per_grad_iteration(x, data['mean_lengths'], run_window=10, title='Mean time steps')

if __name__ == '__main__':
    main()
