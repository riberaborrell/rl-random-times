from sde_agent import SdeAgent
from base_parser import get_base_parser
from figures import MyFigure
from utils_path import get_mc_prediction_dir_path

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
    #agent.set_dir_path('mc-prediction')
    agent.set_dir_path('mc-prediction-es')
    agent.dir_path = get_mc_prediction_dir_path(agent.dir_path, args.gamma, args.n_episodes_lim)

    # initialize hjb solver
    sol_hjb = SolverHJB(
        potential_name='nd_2well',
        n=1,
        alpha=np.ones(1),
        beta=1.,
        h=0.001,
    )

    # load already computed solution
    sol_hjb.load()

    # run mc-learning agent
    if not args.load:

        # preallocate information for all epochs
        agent.n_episodes = args.n_episodes_lim
        agent.preallocate_episodes()

        # set number of averaged episodes 
        agent.n_avg_episodes = args.n_avg_episodes

        # set epsilons
        agent.set_constant_epsilons(args.eps_init)
        #agent.set_glie_epsilons()
        #agent.set_epsilon_parameters(args.eps_init, args.eps_min, args.eps_max, args.eps_decay)

        # set state space and discretize
        agent.set_state_space()
        agent.discretize_state_space(h=0.1)
        agent.discretize_action_space(h=0.1)
        assert agent.state_space_h.shape == sol_hjb.u_opt[::100].shape, ''

        # set deterministic policy
        policy = np.array([
            agent.get_action_idx(sol_hjb.u_opt[::100][idx_state])
            for idx_state, _ in enumerate(agent.state_space_h[:, 0])
        ])

        # mc prediction algorithm
        mc_prediction(agent, policy, args.n_steps_lim)

        # save agent
        agent.save()

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
        agent.plot_value_function(F_hjb=sol_hjb.F[::100])
        #agent.plot_sliced_q_tables()


    # print running avg
    episodes = np.arange(agent.n_episodes)
    for ep in episodes:

        if args.do_report and ep % 1 == 0:
            msg = agent.log_episodes(ep)
            print(msg)

def mc_prediction(agent, policy, n_steps_lim):

    # initialize frequency and q-values table
    agent.initialize_frequency_v_table()
    agent.initialize_v_table()

    # for each episode
    for ep in np.arange(agent.n_episodes):

        # reset environment
        _ = agent.env.reset()
        state = agent.env.state

        # reset trajectory
        agent.reset_states()
        agent.reset_rewards()

        # terminal state flag
        complete = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # get index of the state
            idx_state = agent.get_state_idx(state)

            # choose action following the given policy
            idx_action = policy[idx_state]
            action = agent.action_space_h[idx_action]

            # step dynamics forward
            new_obs, r, complete, _ = agent.env.step(action)
            new_state = agent.env.state

            # save state, actions and reward
            agent.save_state(state)
            agent.save_reward(r)

            # update state
            state = new_state

        # compute return
        agent.compute_discounted_rewards()
        agent.compute_returns()

        # update v values
        n_steps_trajectory = agent.states.shape[0]
        for k in np.arange(n_steps_trajectory):

            # state and its index at step k
            state = agent.states[k]
            idx_state = agent.get_state_idx(state)

            g = agent.returns[k]

            # update frequency and q table
            agent.n_table[idx_state] += 1
            alpha = 1 / agent.n_table[idx_state]
            agent.v_table[idx_state] = agent.v_table[idx_state] \
                                     + alpha * (g - agent.v_table[idx_state])

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
    agent.save_v_table()

if __name__ == '__main__':
    main()
