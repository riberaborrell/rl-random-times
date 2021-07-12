from windygridworld_agent import WindyGridworldAgent
from base_parser import get_base_parser
from figures import MyFigure

import gym
import gym_gridworlds

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('WindyGridworld-v0')

    # initialize Agent
    agent = WindyGridworldAgent(env, args.gamma, logs=args.do_report)

    # get dir path
    agent.set_dir_path('sarsa')

    # run mc learning agent
    if not args.load:

        # set epsilon
        #agent.set_epsilon_parameters(args.eps_init, args.eps_min, args.eps_max, args.eps_decay)

        # set number of averaged episodes 
        agent.n_avg_episodes = 100

        # mc learning
        agent.step_sliced_episodes = args.step_sliced_episodes
        agent.sarsa(args.n_episodes_lim, args.n_steps_lim, args.lr)

        # save agent
        agent.save()

    # load already run agent
    else:
        if not agent.load():
            return

    # do plots
    if args.do_plots:

        # episodes array
        episodes = np.arange(agent.n_episodes)

        # plot sample returns
        fig = MyFigure(agent.dir_path, 'sample_returns')
        y = np.vstack((agent.sample_returns, agent.avg_sample_returns))
        fig.plot_multiple_lines(episodes, y)

        # plot total rewards
        fig = MyFigure(agent.dir_path, 'total_rewards')
        y = np.vstack((agent.total_rewards, agent.avg_total_rewards))
        fig.plot_multiple_lines(episodes, y)

        # plot time steps
        fig = MyFigure(agent.dir_path, 'time_steps')
        fig.plot_one_line(episodes, agent.time_steps)

        # plot epsilons
        fig = MyFigure(agent.dir_path, 'epsilons')
        fig.plot_one_line(episodes, agent.epsilons)

        # plot policies
        agent.compute_policy_table()
        fig = MyFigure(agent.dir_path, 'policy_table')
        fig.axes[0].imshow(agent.policy_table, origin='lower')
        fig.savefig(fig.file_path)


    if args.do_report:
        for ep in np.arange(agent.n_episodes):
            # print running avg
            if ep % 1 == 0:
                msg = agent.log_episodes(ep)
                print(msg)


if __name__ == '__main__':
    main()
