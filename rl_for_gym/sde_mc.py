from sde_agent import SdeAgent
from base_parser import get_base_parser
from figures import MyFigure

import gym
import gym_sde

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
    env = gym.make('sde-v0')
    env.random_x_init = True

    # initialize Agent
    agent = SdeAgent(env, args.gamma, logs=args.do_report)

    # get dir path
    agent.set_dir_path('mc-learning')

    # run mc-learning agent
    if not args.load:

        # preallocate information for all epochs
        agent.n_episodes = args.n_episodes_lim
        agent.preallocate_episodes()

        # set number of averaged episodes 
        agent.n_avg_episodes = args.n_avg_episodes

        # set epsilons
        agent.set_glie_epsilons()
        #agent.set_epsilon_parameters(args.eps_init, args.eps_min, args.eps_max, args.eps_decay)

        # q-learning algorithm
        agent.mc_learning(args.n_steps_lim, args.alpha)

        # save agent
        #agent.step_sliced_episodes = args.step_sliced_episodes

    # load already run agent
    else:
        if not agent.load():
            return

        # set state space and discretize
        agent.set_state_space()
        agent.discretize_state_space()
        agent.discretize_action_space()

    # do plots
    if args.do_plots:
        agent.episodes = np.arange(agent.n_episodes)
        agent.plot_total_rewards()
        agent.plot_time_steps()
        agent.plot_epsilons()
        agent.plot_control()
        #agent.plot_sliced_q_tables()


    # print running avg
    episodes = np.arange(agent.n_episodes)
    for ep in episodes:

        if args.do_report and ep % 1 == 0:
            msg = agent.log_episodes(ep)
            print(msg)


if __name__ == '__main__':
    main()
