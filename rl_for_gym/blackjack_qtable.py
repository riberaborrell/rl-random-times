from blackjack_agent import BlackjackAgent
from base_parser import get_base_parser

import gym
import numpy as np


def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser


def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('Blackjack-v0')

    # initialize Agent
    agent = BlackjackAgent(env, args.gamma, logs=args.do_report)

    # get dir path
    agent.set_dir_path('mc-learning')

    # load already run agent
    agent.load()
    q_table = agent.last_q_table

    # preallocate information for all epochs
    agent.n_episodes = args.n_episodes_lim
    agent.preallocate_episodes()

    # set number of averaged episodes 
    agent.n_avg_episodes = args.n_avg_episodes

    # different trajectories
    for ep in np.arange(agent.n_episodes):

        # reset environment
        observation = agent.env.reset()
        state = agent.get_state(observation)

        # reset trajectory
        agent.reset_rewards()

        # terminal state flag
        complete = False

        for k in range(args.n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # take action with highest q-value 
            action = np.argmax(q_table[state])

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
