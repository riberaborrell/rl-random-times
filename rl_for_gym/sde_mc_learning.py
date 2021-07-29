from rl_for_gym.sde_agent import SdeAgent
from rl_for_gym.base_parser import get_base_parser
from rl_for_gym.utils_path import get_mc_dir_path

from mds.langevin_nd_hjb_solver import SolverHJB

import gym
import gym_sde

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    parser.add_argument(
        '--es',
        dest='explorable_starts',
        action='store_true',
        help='the initial point of the trajectory is uniform sampled.',
    )
    parser.add_argument(
        '--constant-alpha',
        dest='constant_alpha',
        action='store_true',
        help='the step size parameter is given by alpha',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('sde-v0')

    # explorable starts
    if args.explorable_starts:
        env.random_x_init = True

    # initialize Agent
    agent = SdeAgent(env, args.gamma, logs=args.do_report)

    # get dir path
    agent.set_dir_path()
    agent.dir_path = get_mc_dir_path(agent.dir_path, args.explorable_starts,
                                     args.constant_alpha, args.alpha,
                                     args.gamma, 'const', args.n_episodes_lim)

    # run mc-learning agent
    if not args.load:

        # preallocate information for all epochs
        agent.n_episodes = args.n_episodes_lim
        agent.preallocate_episodes()

        # set number of averaged episodes 
        agent.n_avg_episodes = args.n_avg_episodes

        # set state space and discretize
        agent.set_state_space()
        agent.discretize_state_space(h=0.05)
        agent.discretize_action_space(h=0.05)

        # set epsilons
        agent.set_constant_epsilons(args.eps_init)
        #agent.set_glie_epsilons()
        #agent.set_epsilon_parameters(args.eps_init, args.eps_min, args.eps_max, args.eps_decay)

        # mc learning algorithm
        if not args.constant_alpha:
            mc_learning(agent, args.n_steps_lim)
        else:
            mc_learning(agent, args.n_steps_lim, args.alpha)

        # save agent
        agent.save()

    # load already run agent
    else:
        if not agent.load():
            return

        # set state space and discretize
        agent.set_state_space()
        agent.discretize_state_space(h=0.05)
        agent.discretize_action_space(h=0.05)

    # do plots
    if args.do_plots:
        agent.episodes = np.arange(agent.n_episodes)
        agent.plot_total_rewards()
        agent.plot_time_steps()
        agent.plot_epsilons()
        agent.plot_frequency_table()
        agent.plot_q_table()
        agent.plot_control()
        #agent.plot_sliced_q_tables()

    # print running avg if load
    if not args.load:
        return

    episodes = np.arange(agent.n_episodes)
    for ep in episodes:

        if args.do_report and ep % 1 == 0:
            msg = agent.log_episodes(ep)
            print(msg)

def mc_learning(agent, n_steps_lim, alpha=None):

    # initialize frequency and q-values table
    agent.initialize_frequency_table()
    agent.initialize_q_table()

    # for each episode
    for ep in np.arange(agent.n_episodes):

        # reset environment
        _ = agent.env.reset()
        state = agent.env.state

        # reset trajectory
        agent.reset_trajectory()

        # terminal state flag
        complete = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # get index of the state
            idx_state = agent.get_state_idx(state)

            # choose action following epsilon greedy policy
            idx_action, action = agent.get_epsilon_greedy_action(ep, idx_state)

            # step dynamics forward
            new_obs, r, complete, _ = agent.env.step(action)
            new_state = agent.env.state

            # save state, actions and reward
            agent.save_trajectory(state, action, r)

            # update state
            state = new_state

        # compute return
        agent.compute_discounted_rewards()
        agent.compute_returns()

        # update q values
        n_steps_trajectory = agent.states.shape[0]
        for k in np.arange(n_steps_trajectory):

            # state and its index at step k
            state = agent.states[k]
            idx_state = agent.get_state_idx(state)

            # action and its index at step k
            action = agent.actions[k]
            idx_action = agent.get_action_idx(state)

            # state-action index
            idx = (idx_state, idx_action)
            g = agent.returns[k]

            # update frequency table
            agent.n_table[idx] += 1

            # set learning rate
            if alpha is None:
                alpha = 1 / agent.n_table[idx]

            # update q table
            agent.q_table[idx] += alpha * (g - agent.q_table[idx])

        # save time steps
        agent.save_episode(ep, k)

        # logs
        if agent.logs and ep % 100 == 0:
            msg = agent.log_episodes(ep)
            print(msg)


    # update npz dict
    agent.update_npz_dict_agent()

    # save frequency and q-value last tables
    agent.save_frequency_table()
    agent.save_q_table()

if __name__ == '__main__':
    main()
