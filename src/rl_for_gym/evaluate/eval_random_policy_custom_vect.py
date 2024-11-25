import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.path import load_data, save_data, get_dir_path
from rl_for_gym.utils.plots import plot_y_per_episode

def random_policy_vect(env, gamma: float = 1., batch_size: int = 100, n_steps_lim: int = 10**6,
                       truncate=True, log_freq: int = 10, seed=None, load=False):

    # get dir path
    dir_path = get_dir_path(env.spec.id, algorithm_name='random')

    # load results
    if load:
        return load_data(dir_path)

    # preallocate arrays
    returns = np.empty(batch_size, dtype=np.float32)
    time_steps = np.empty(batch_size, dtype=np.int32)

    # reset environment
    options = {'batch_size': batch_size}
    obs, info = env.reset(seed=seed, options=options)

    # reset total rewards
    rewards = np.empty((batch_size, 0))

    # terminated flags
    been_terminated = np.full((batch_size,), False)
    new_terminated = np.full((batch_size,), False)
    done = been_terminated

    while not done.all():

        # take a random actions
        action = env.unwrapped.action_space_vect.sample()

        # step dynamics forward
        obs, r, terminated, truncated, infos = env.step(action)

        # check if truncattion is allowed
        truncated = False if not truncate else truncated

        # update terminated flags
        new_terminated = terminated & ~been_terminated
        been_terminated = terminated | been_terminated

        # done flags
        done = np.logical_or(been_terminated, truncated)

        # save reward
        rewards = np.hstack((rewards, np.expand_dims(r, axis=1)))

        # save time steps
        if new_terminated.any():
            returns[new_terminated] = np.sum(rewards[new_terminated], axis=1)
            time_steps[new_terminated] = env._elapsed_steps

        # interrupt if all trajectories have reached a terminal state or have been truncated 
        if done.all():
            returns[~been_terminated] = np.sum(rewards[~been_terminated], axis=1)
            time_steps[~been_terminated] = env._elapsed_steps
            break

    print('Mean return: {:.1f}'.format(np.mean(returns)))
    print('Mean time steps: {:.1f}'.format(np.mean(time_steps)))

    data = {
        'returns': returns,
        'time_steps': time_steps,
    }
    save_data(data, dir_path)
    return True, data


def main():
    args = get_base_parser().parse_args()

    # create gym env
    env = gym.make(args.env_id, max_episode_steps=args.n_steps_lim, is_vectorized=True)

    # run random policy vectorized
    succ, data = random_policy_vect(
        env,
        batch_size=args.n_episodes,
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

if __name__ == '__main__':
    main()
