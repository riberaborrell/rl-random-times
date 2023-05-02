import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt

from rl_for_gym.base_parser import get_base_parser
from rl_for_gym.reinforce_discrete import reinforce

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # restrict to tested environments
    assert args.env_id in ['Acrobot-v1', 'CartPole-v1']

    # create gym env 
    if args.render:
        env = gym.make(args.env_id, render_mode='human')
    else:
        env = gym.make(args.env_id)

    # run reinforce
    returns, time_steps, model = reinforce(
        env,
        gamma=args.gamma,
        lr=args.lr,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        render=args.render,
    )

    window = args.batch_size

    # plot returns
    smoothed_returns = [
        np.mean(returns[i-window:i+1]) if i > window
        else np.mean(returns[:i+1]) for i in range(len(returns))
    ]
    plt.figure(figsize=(12, 8))
    plt.plot(returns)
    plt.plot(smoothed_returns)
    plt.ylabel('Total Returns')
    plt.xlabel('Episodes')
    plt.show()

    # plot time steps
    smoothed_time_steps = [
        np.mean(time_steps[i-window:i+1]) if i > window
        else np.mean(time_steps[:i+1]) for i in range(len(time_steps))
    ]
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps)
    plt.plot(smoothed_time_steps)
    plt.ylabel('Total Time steps')
    plt.xlabel('Episodes')
    plt.show()


if __name__ == '__main__':
    main()
