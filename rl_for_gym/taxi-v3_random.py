from base_parser import get_base_parser
from agent import Agent
from plots import Plot

import numpy as np
import gym

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('Taxi-v3')

    # initialize Agent
    agent = Agent(env, args.gamma)

    # set dir path
    agent.set_dir_path('random')

    # preallocate information for all epochs
    agent.preallocate_episodes()

    # different trajectories
    for ep in np.arange(args.n_episodes_lim):

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

            # render observation
            if args.do_render:
                agent.env.render()

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
        agent.save_episode(time_steps=k)

    # save number of episodes
    agent.n_episodes = ep + 1

    if args.do_plots:

        # plot sample returns
        plt = Plot(agent.dir_path, 'sample_returns')
        plt.one_line_plot(agent.n_episodes, agent.sample_returns)

        # plot total rewards
        plt = Plot(agent.dir_path, 'total_rewards')
        plt.one_line_plot(agent.n_episodes, agent.total_rewards)

        # plot time steps
        plt = Plot(agent.dir_path, 'time_steps')
        plt.one_line_plot(agent.n_episodes, agent.time_steps)


    if args.do_report:
        for ep in np.arange(agent.n_episodes):

            # print running avg
            if ep % 1 == 0:
                msg = agent.log_episodes(ep)
                print(msg)


if __name__ == '__main__':
    main()
