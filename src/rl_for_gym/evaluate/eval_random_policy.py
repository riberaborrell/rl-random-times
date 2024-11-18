import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.numeric import discount_cumsum
from rl_for_gym.utils.path import load_data, save_data, get_dir_path

def random_policy(env, gamma: float = 1., n_episodes: int = 100, n_steps_lim: int = 10**5,
                  log_freq: int = 10, seed=None, load=False):

    # get dir path
    dir_path = get_dir_path(env, algorithm_name='random')

    # load results
    if load:
        return load_data(dir_path)

    # preallocate arrays
    returns = np.empty(n_episodes)
    time_steps = np.empty(n_episodes, dtype=np.int32)

    # sample trajectories
    for ep in np.arange(n_episodes):

        # reset environment
        obs, info = env.reset(seed=seed)

        # reset rewards
        rewards = np.empty(0)

        for k in range(1, n_steps_lim+1):

            # take a random action
            action = env.action_space.sample()

            # step dynamics forward
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # save reward
            rewards = np.append(rewards, r)

            # interrupt if we are in a terminal state
            if done:
                break

        # compute return at each time step
        episode_returns = discount_cumsum(rewards, gamma)

        # save return and time steps
        returns[ep] = episode_returns[0]
        time_steps[ep] = k

        if (ep + 1) % log_freq == 0:
            msg = 'ep: {:3d}, time steps: {:4d}, return {:2.2f}' \
                  ''.format(
                        ep+1,
                        time_steps[ep],
                        returns[ep],
                      )
            print(msg)

    data = {
        'returns': returns,
        'time_steps': time_steps,
    }
    save_data(data, dir_path)
    return True, data

def main():
    args = get_base_parser().parse_args()

    # create gym env 
    if args.render:
        env = gym.make(args.env_id, render_mode='human')
    else:
        env = gym.make(args.env_id)

    # run random policy
    succ, data = random_policy(
        env,
        n_episodes=args.n_episodes,
        seed=args.seed,
        log_freq=args.log_freq,
        load=args.load,
    )
    env.close()

    # do plots
    if not args.plot or not succ:
        return

    # unpack data
    returns = data['returns']
    time_steps = data['time_steps']

    # fht histogram
    x = np.arange(100)
    counts, bins = np.histogram(time_steps, bins=x, density=True)
    fig, ax = plt.subplots()
    ax.set_xlabel('m')
    ax.hist(bins[:-1], bins, weights=counts, alpha=0.5, color='tab:orange', label=r'histogram')
    ax.legend()
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
