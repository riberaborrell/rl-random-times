from base_parser import get_base_parser
from sde_agent import SdeAgent
from figures import MyFigure

import gym
import gym_sde
import numpy as np


def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('sde-v0')
    env.random_x_init = True

    # initialize Agent
    agent = SdeAgent(env, args.gamma)

    # set dir path
    agent.set_dir_path('random')

    # set seed
    #if args.seed:
    #    agent.env.seed(args.seed)

    # preallocate information for all epochs
    agent.n_episodes = args.n_episodes_lim
    agent.preallocate_episodes()

    # set number of averaged episodes 
    agent.n_avg_episodes = args.n_avg_episodes

    # different trajectories
    for ep in np.arange(agent.n_episodes):

        # reset environment
        obs = agent.env.reset()

        # reset trajectory
        agent.reset_rewards()

        # terminal state flag
        complete = False

        for k in range(args.n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # render observation
            #if args.do_render:
            #    agent.env.render()

            # take a random action
            action = agent.env.action_space.sample()

            # step dynamics forward
            obs, r, complete, _ = agent.env.step(action)

            # save reward
            agent.save_reward(r)

        # close env
        agent.env.close()

        # compute return
        agent.compute_discounted_rewards()
        agent.compute_returns()

        # save time steps
        agent.save_episode(ep, k)

        if args.do_report and ep % 1 == 0:
            msg = agent.log_episodes(ep)
            print(msg)

    # do plots
    if args.do_plots:
        agent.episodes = np.arange(agent.n_episodes)
        agent.plot_total_rewards()
        agent.plot_time_steps()

if __name__ == '__main__':
    main()
