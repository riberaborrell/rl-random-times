from windygridworld_agent import WindyGridworldAgent
from base_parser import get_base_parser

import gym
import gym_gridworlds

import numpy as np

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
    agent.set_dir_path('sarsa-lambda')

    # run mc learning agent
    if not args.load:

        # set number of averaged episodes 
        agent.n_avg_episodes = 100

        # sarsa algorithm
        agent.sarsa_lambda(args.n_episodes_lim, args.n_steps_lim, args.lr, args.lam)

        # save agent
        agent.save()

    # load already run agent
    else:
        if not agent.load():
            return

    # do plots
    if args.do_plots:
        agent.episodes = np.arange(agent.n_episodes)
        agent.plot_sample_returns()
        agent.plot_total_rewards()
        agent.plot_time_steps()
        agent.plot_epsilons()
        agent.plot_frequency()
        agent.plot_policy()

    if args.do_report:
        for ep in np.arange(agent.n_episodes):
            # print running avg
            if ep % 1 == 0:
                msg = agent.log_episodes(ep)
                print(msg)


if __name__ == '__main__':
    main()
