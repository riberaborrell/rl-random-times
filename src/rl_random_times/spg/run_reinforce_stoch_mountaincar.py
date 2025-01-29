import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from rl_random_times.base_parser import get_base_parser
from rl_random_times.reinforce_discrete import reinforce

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # initialize environment 
    from gymnasium.wrappers import TimeLimit
    env = gym.make('MountainCar-v0')
    env = TimeLimit(env, max_episode_steps=1000)

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
    # plot action probability distributions

    # get flat discretized observation space
    h_x, h_y = 0.1, 0.001
    slice_x = slice(env.observation_space.low[0], env.observation_space.high[0]+h_x, h_x)
    slice_y = slice(env.observation_space.low[1], env.observation_space.high[1]+h_y, h_y)
    observation_space_h = np.moveaxis(np.mgrid[slice_x, slice_y], 0, -1)
    Nx = observation_space_h.shape[0]
    Ny = observation_space_h.shape[1]
    N = Nx*Ny
    observation_space_h_flat = observation_space_h.reshape(N, 2)

    # compute action probability distributions
    actions_probs = model.forward(observation_space_h_flat).detach().numpy().reshape(Nx, Ny, 3)
    most_prob_actions = np.argmax(actions_probs, axis=2)

    extent = env.observation_space.low[0], env.observation_space.high[0], \
             env.observation_space.low[1], env.observation_space.high[1]
    plt.imshow(most_prob_actions, extent=extent)
    #cmap = cm.get_cmap('Paired_r', 3)
    #plt.imshow(most_prob_actions, extent=extent, cmap=cmap)
    plt.colorbar()
    plt.show()

    return


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
