import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.numeric import discount_cumsum
from rl_for_gym.utils.path import load_data, save_data, get_dir_path
from rl_for_gym.utils.plots import plot_y_per_episode

def random_policy(env, gamma: float = 1., n_episodes: int = 100,
                  truncate: bool = True, log_freq: int = 10, seed=None, load=False):

    # get dir path
    dir_path = get_dir_path(env.spec.id, algorithm_name='random')

    # load results
    if load:
        return load_data(dir_path)

    # set seed
    if seed is not None:
        env.action_space.seed(seed)

    # preallocate arrays
    returns = np.zeros(n_episodes)
    time_steps = np.zeros(n_episodes, dtype=np.int32)

    # sample trajectories
    for ep in np.arange(n_episodes):

        # reset environment
        obs, info = env.reset(seed=seed) if ep == 0 else env.reset()

        # reset rewards
        rewards = np.empty(0)

        # done flag
        done = False

        while not done:

            # take a random action
            action = env.action_space.sample()

            # step dynamics forward
            obs, r, terminated, truncated, info = env.step(action)
            truncated = False if not truncate else truncated
            done = terminated or truncated

            # save reward
            rewards = np.append(rewards, r)

        # compute return at each time step
        episode_returns = discount_cumsum(rewards, gamma)

        # save return and time steps
        returns[ep] = episode_returns[0]
        time_steps[ep] = env._elapsed_steps

        if (ep + 1) % log_freq == 0:
            msg = 'ep: {:3d}, time steps: {:4d}, return {:2.2f}' \
                  ''.format(
                        ep+1,
                        time_steps[ep],
                        returns[ep],
                      )
            print(msg)

    print('Mean return: {:.3f}'.format(np.mean(returns)))
    print('Mean time steps: {:.3f}'.format(np.mean(time_steps)))

    data = {
        'returns': returns,
        'time_steps': time_steps,
    }
    save_data(data, dir_path)
    return True, data

def main():
    args = get_base_parser().parse_args()

    # set render mode
    render_mode = 'human' if args.render else None

    # create gym env 
    env = gym.make(args.env_id, max_episode_steps=args.n_steps_lim, render_mode=render_mode)

    # run random policy
    succ, data = random_policy(
        env,
        n_episodes=args.n_episodes,
        seed=args.seed,
        truncate=args.truncate,
        log_freq=args.log_freq,
        load=args.load,
    )
    env.close()

    # do plots
    if not args.plot or not succ:
        return

    # plot returns and time steps
    x = np.arange(args.n_episodes)
    plot_y_per_episode(x, data['returns'], run_window=100, title='Returns', legend=True)
    plot_y_per_episode(x, data['time_steps'], run_window=100, title='Time steps')

    # fht histogram
    x = np.arange(100)
    counts, bins = np.histogram(data['time_steps'], bins=x, density=True)
    fig, ax = plt.subplots()
    ax.set_xlabel('m')
    ax.hist(bins[:-1], bins, weights=counts, alpha=0.5, color='tab:orange', label=r'histogram')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
