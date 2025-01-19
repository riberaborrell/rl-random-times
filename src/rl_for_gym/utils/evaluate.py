import time

import numpy as np
import torch

from rl_for_gym.utils.numeric import discount_cumsum
from rl_for_gym.utils.path import load_data, save_data, get_dir_path

def simulate_random_policy_episode(env, seed=1, log_freq=None, time_sleep=None):

    # reset environment
    obs, info = env.reset(seed=seed)

    done = False
    while not done:

        # slow down to visualize
        if time_sleep is not None:
            time.sleep(time_sleep)

        # take a random action
        action = env.action_space.sample()

        # step dynamics forward
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # log
        k = env.get_wrapper_attr('_elapsed_steps')
        if log_freq is not None and (k + 1) % log_freq == 0:
            print(k, obs, r, done, truncated)

    env.close()

def simulate_learnt_policy_episode(env, policy, seed=1, log_freq=None, time_sleep=None):

    # reset environment
    obs, info = env.reset(seed=seed)

    done = False
    while not done:

        # slow down to visualize
        if time_sleep is not None:
            time.sleep(time_sleep)

        # take a random action
        action, _ = policy.sample_action(torch.tensor(obs, dtype=torch.float32))

        # step dynamics forward
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # log
        k = env._elapsed_steps
        if log_freq is not None and (k + 1) % log_freq == 0:
            print(k, obs, r, done, truncated)

    env.close()

def compute_std_and_re(mean: float, var: float):
    std = np.sqrt(var)
    return std, std / mean

def log_array_stats(mean, var, std=None, re=None, name: str = 'x'):
    if std is None or re is None:
        std, re = compute_std_and_re(mean, var)
    print('{}: mean: {:2.2e}, re: {:.3f}'.format(name, mean, re))

def log_stats(returns, time_steps, ct):
    log_array_stats(returns.mean(), returns.var(), name='returns')
    log_array_stats(time_steps.mean(), time_steps.var(), name='time steps')
    print('ct: {:2.1e}'.format(ct)) if ct is not None else None


def eval_random_policy(env, gamma: float = 1., n_episodes: int = 100,
                      truncate: bool = True, log_freq: int = 10, seed=None, load=False):

    # get dir path
    dir_path = get_dir_path(env.unwrapped.spec.id, algorithm_name='random')

    # load results
    if load:
        return load_data(dir_path)

    # set seed
    if seed is not None:
        env.action_space.seed(seed)

    # start timer
    initial_t = time.perf_counter()

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
            #action = np.zeros(env.action_space.shape[0])

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
        time_steps[ep] = env.get_wrapper_attr('_elapsed_steps')

        if (ep + 1) % log_freq == 0:
            msg = 'ep: {:3d}, time steps: {:4d}, return {:2.2f}' \
                  ''.format(
                        ep+1,
                        time_steps[ep],
                        returns[ep],
                      )
            print(msg)

    # close timer
    ct = time.perf_counter() - initial_t

    log_stats(returns, time_steps, ct)

    data = {
        'returns': returns,
        'time_steps': time_steps,
    }
    save_data(data, dir_path)
    return True, data


def eval_random_policy_vect_sync(env, gamma: float = 1., truncate=True,
                                 log_freq: int = 10, seed=None, load=False):

    # get dir path
    dir_path = get_dir_path(env.spec.id, algorithm_name='random')

    # load results
    if load:
        return load_data(dir_path)

    # set seed
    if seed is not None:
        env.action_space.seed(seed)

    # start timer
    initial_t = time.perf_counter()

    # reset environment
    obs, info = env.reset(seed=seed)

    # preallocate time steps
    batch_size = env.num_envs
    time_steps = np.empty(batch_size, dtype=np.int32)

    # reset rewards
    rewards = []

    # terminated flags
    been_terminated = np.full((batch_size,), False)
    new_terminated = np.full((batch_size,), False)
    done = new_terminated

    while not done.all():

        # take a random actions
        actions = env.action_space.sample()
        #actions = np.zeros(env.unwrapped.action_space.shape)

        # step dynamics forward
        obs, r, terminated, truncated, infos = env.step(actions)

        # check if truncattion is allowed
        if not truncate:
            truncated[:] = False

        # update terminated flags
        new_terminated = terminated & ~been_terminated
        been_terminated = terminated | been_terminated

        # done flags
        done = np.logical_or(been_terminated, truncated)

        # save rewards
        rewards.append(r)

        # save time steps
        if new_terminated.any() or truncated.any():
            for i in np.where(new_terminated | truncated)[0]:
                time_steps[i] = env.envs[i]._elapsed_steps

        # interrupt if all trajectories have reached a terminal state or have been truncated 
        if done.all():
            break

    # compute returns
    #TODO: generalize to discounted returns
    returns = np.array([np.stack(rewards)[:time_steps[i], i].sum() for i in range(batch_size)])

    # close timer
    ct = time.perf_counter() - initial_t

    # log statistics
    log_stats(returns, time_steps, ct)

    data = {
        'returns': returns,
        'time_steps': time_steps,
    }
    save_data(data, dir_path)
    return True, data

def eval_random_policy_envpool_sync(env, env_id: str, gamma: float = 1.,
                       truncate=True, log_freq: int = 10, load=False):

    # get dir path
    dir_path = get_dir_path(env_id, algorithm_name='random')

    # load results
    if load:
        return load_data(dir_path)

    # set seed
    env.action_space.seed(env.spec.config.seed)

    # start timer
    initial_t = time.perf_counter()

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

    # close timer
    ct = time.perf_counter() - initial_t

    # log statistics
    log_stats(returns, time_steps, ct)

    data = {
        'returns': returns,
        'time_steps': time_steps,
    }
    save_data(data, dir_path)
    return True, data


def eval_random_policy_custom_vect(env, gamma: float = 1., batch_size: int = 100, n_steps_lim: int = 10**6,
                                   truncate=True, log_freq: int = 10, seed=None, load=False):

    # get dir path
    dir_path = get_dir_path(env.spec.id, algorithm_name='random')

    # load results
    if load:
        return load_data(dir_path)

    # set seed
    if seed is not None:
        env.action_space.seed(seed)

    # start timer
    initial_t = time.perf_counter()

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
        #action = np.zeros(batch_size)

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

    # close timer
    ct = time.perf_counter() - initial_t

    log_stats(returns, time_steps, ct)

    data = {
        'returns': returns,
        'time_steps': time_steps,
    }
    save_data(data, dir_path)
    return True, data


