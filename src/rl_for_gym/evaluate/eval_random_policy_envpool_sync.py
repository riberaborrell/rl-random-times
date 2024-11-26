import envpool
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.path import load_data, save_data, get_dir_path
from rl_for_gym.utils.plots import plot_y_per_episode

def random_policy_sync(env, env_id: str, gamma: float = 1.,
                       truncate=True, log_freq: int = 10, load=False):

    # get dir path
    dir_path = get_dir_path(env_id, algorithm_name='random')

    # load results
    if load:
        return load_data(dir_path)

    # set seed
    env.action_space.seed(env.spec.config.seed)

    # preallocate arrays
    batch_size = env.spec.config.num_envs
    returns = np.empty(batch_size, dtype=np.float32)
    time_steps = np.empty(batch_size, dtype=np.int32)

    # reset environment
    obs, info = env.reset()

    # reset rewards
    rewards = np.empty((batch_size, 0))

    # terminated and done flags
    been_terminated = np.full((batch_size,), False)
    new_terminated = np.full((batch_size,), False)
    done = been_terminated

    # time step counter
    k = 1

    while not done.all():

        # take a random actions
        action = np.stack([env.action_space.sample() for _ in range(batch_size)])

        # step dynamics forward
        obs, r, terminated, truncated, infos = env.step(action)

        # check if truncattion is allowed
        if not truncate:
            truncated[:] = False

        # update terminated flags
        new_terminated = terminated & ~been_terminated
        been_terminated = terminated | been_terminated

        # done flags
        done = np.logical_or(been_terminated, truncated)

        # save reward
        rewards = np.hstack((rewards, np.expand_dims(r, axis=1)))

        # save time steps
        if new_terminated.any() or truncated.any():
            idx = new_terminated | truncated
            returns[idx] = np.sum(rewards[idx], axis=1)
            time_steps[idx] = k

        # interrupt if all trajectories have reached a terminal state or have been truncated 
        if done.all():
            break

        # increment time step counter
        k += 1

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

    # create gym env
    if args.n_steps_lim is None:
        env = envpool.make_gymnasium(args.env_id, num_envs=args.n_envs, seed=args.seed)
    else:
        env = envpool.make_gymnasium(args.env_id, num_envs=args.n_envs,
                                     seed=args.seed, max_episode_steps=args.n_steps_lim)

    # run random policy vectorized
    succ, data = random_policy_sync(
        env,
        args.env_id,
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
