import time

import numpy as np
import torch

from rl_for_gym.utils.numeric import discount_cumsum
from rl_for_gym.utils.path import load_data, save_data, get_dir_path

def simulate_random_policy_episode(env, args):

    # reset environment
    obs, info = env.reset(seed=args.seed)

    for k in range(args.n_steps_lim):

        # take a random action
        action = env.action_space.sample()

        # step dynamics forward
        obs, r, terminated, truncated, info = env.step(action)
        truncated = False if not args.truncate else truncated
        done = terminated or truncated

        # log
        if (k + 1) % args.log_freq == 0:
            print(k, obs, r, done, truncated)

        if done:
            break

    env.close()

def simulate_learnt_policy_episode(env, policy, args):

    # reset environment
    obs, info = env.reset(seed=args.seed)

    for k in range(args.n_steps_lim):

        # slow down to visualize
        time.sleep(0.1)

        # take a random action
        action, _ = policy.sample_action(torch.tensor(obs, dtype=torch.float32))

        # step dynamics forward
        obs, r, terminated, truncated, info = env.step(action)
        truncated = False if not args.truncate else truncated
        done = terminated or truncated

        # log
        if (k + 1) % args.log_freq == 0:
            print(k, obs, r, done, truncated)

        if done:
            break

    env.close()


