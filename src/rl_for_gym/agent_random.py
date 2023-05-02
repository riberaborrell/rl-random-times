from base_parser import get_base_parser
from utils_numeric import discount_cumsum

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    if args.render:
        env = gym.make(args.env_id, render_mode='human')
    else:
        env = gym.make(args.env_id)

    # preallocate arrays
    returns = np.empty(args.n_episodes)
    time_steps = np.empty(args.n_episodes, dtype=np.int32)

    # sample trajectories
    for ep in np.arange(args.n_episodes):

        # reset environment
        obs, info = env.reset(seed=args.seed)

        # reset rewards
        rewards = np.empty(0)

        for k in range(args.n_steps_lim):

            # take a random action
            action = env.action_space.sample()

            # step dynamics forward
            obs, r, done, truncated, info = env.step(action)

            # save reward
            rewards = np.append(rewards, r)

            # interrupt if we are in a terminal state
            if done or truncated:
                break

        # compute return at each time step
        episode_returns = discount_cumsum(rewards, args.gamma)

        # save return and time steps
        returns[ep] = episode_returns[0]
        time_steps[ep] = k

        if (ep + 1) % args.log_interval == 0:
            msg = 'ep: {:3d}, time steps: {:4d}, return {:2.2f}' \
                  ''.format(
                        ep+1,
                        time_steps[ep],
                        returns[ep],
                      )
            print(msg)

    env.close()

    # do plots
    if not args.plot:
        return

if __name__ == '__main__':
    main()
