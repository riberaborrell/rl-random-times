from base_parser import get_base_parser
from blackjack_agent import BlackjackAgent
from figures import MyFigure

import numpy as np
import gym

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('Blackjack-v0')

    # initialize Agent
    agent = BlackjackAgent(env, args.gamma)

    # set dir path
    agent.set_dir_path('random')

    # preallocate information for all epochs
    agent.n_episodes = args.n_episodes_lim
    agent.preallocate_episodes()

    # set number of averaged episodes 
    agent.n_avg_episodes = args.n_avg_episodes

    # different trajectories
    for ep in np.arange(agent.n_episodes):

        # reset environment
        agent.env.reset()

        # reset trajectory
        agent.reset_rewards()

        # terminal state flag
        complete = False

        for k in range(args.n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # take a random action
            action = env.action_space.sample()

            # step dynamics forward
            _, r, complete, _ = env.step(action)

            # save reward
            agent.save_reward(r)

        # close env
        agent.env.close()

        # compute return
        agent.compute_discounted_rewards()
        agent.compute_returns()

        # save time steps
        agent.save_episode(ep, k)

        # logs
        if args.do_report:
            msg = agent.log_episodes(ep)
            print(msg)

    # do plots
    if args.do_plots:
        episodes = np.arange(agent.n_episodes)
        agent.episodes = np.arange(agent.n_episodes)
        agent.plot_sample_returns()
        agent.plot_total_rewards()
        agent.plot_time_steps()

if __name__ == '__main__':
    main()
