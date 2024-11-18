import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.path import load_data, save_data, get_dir_path

def random_policy_vect(envs, gamma: float = 1., n_episodes: int = 100, n_steps_lim: int = 10**5,
                       truncate=True, log_freq: int = 10, seed=None, load=False):

    # get dir path
    dir_path = get_dir_path(envs, algorithm_name='random')

    # load results
    if load:
        return load_data(dir_path)

    # preallocate arrays
    batch_size = envs.num_envs
    returns = np.empty(batch_size, dtype=np.float32)
    time_steps = np.empty(batch_size, dtype=np.int32)

    # reset environment
    obs, info = envs.reset(seed=seed)

    # reset rewards
    rewards = np.empty((batch_size, 0))

    # terminated flags
    been_terminated = np.full((batch_size,), False)
    new_terminated = np.full((batch_size,), False)

    for k in range(n_steps_lim):

        # take a random actions
        actions = envs.action_space.sample()

        # step dynamics forward
        obs, r, terminated, truncated, infos = envs.step(actions)

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
        if new_terminated.shape[0] != 0:
            returns[new_terminated] = np.sum(rewards[new_terminated], axis=1)
            time_steps[new_terminated] = k

        # interrupt if all trajectories have reached a terminal state or have been truncated 
        if done.all():
            break

    print('Mean return: ', np.mean(returns))
    print('Mean time steps: ', np.mean(time_steps))

    data = {
        'returns': returns,
        'time_steps': time_steps,
    }
    save_data(data, dir_path)
    return True, data


def main():
    args = get_base_parser().parse_args()

    # create gym envs 
    envs = gym.make_vec(args.env_id, num_envs=args.n_episodes, vectorization_mode="async")

    # run random policy vectorized
    succ, data = random_policy_vect(
        envs,
        seed=args.seed,
        truncate=args.truncate,
        log_freq=args.log_freq,
        load=args.load,
    )
    envs.close()


    # do plots
    if not args.plot or not succ:
        return

if __name__ == '__main__':
    main()
