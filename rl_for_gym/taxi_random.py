from base_parser import get_base_parser
from agent import Agent
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
    env = gym.make('Taxi-v3')
    breakpoint()

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

        # logs
        if args.do_report:
            msg = agent.log_episodes(ep)
            print(msg)

    # save number of episodes
    agent.n_episodes = ep + 1

    if args.do_plots:

        # episodes array
        episodes = np.arange(agent.n_episodes)

        # plot sample returns
        fig = MyFigure(agent.dir_path, 'sample_returns')
        fig.plot_one_line(episodes, agent.sample_returns)

        # plot total rewards
        fig = MyFigure(agent.dir_path, 'total_rewards')
        fig.plot_one_line(episodes, agent.total_rewards)

        # plot time steps
        fig = MyFigure(agent.dir_path, 'time_steps')
        fig.plot_one_line(episodes, agent.time_steps)


if __name__ == '__main__':
    main()
