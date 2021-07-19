from sde_agent import SdeAgent
from base_parser import get_base_parser
from utils_path import get_qlearning_dir_path

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

    # initialize Agent
    agent = SdeAgent(env, args.gamma, logs=args.do_report)

    # get dir path
    agent.set_dir_path('q-learning')
    agent.dir_path = get_qlearning_dir_path(agent.dir_path, 'glie', args.alpha,
                                            args.n_episodes_lim)

    # run q-learning agent
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
        q_learning(agent, args.n_steps_lim, args.alpha)

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
        agent.plot_epsilons()
        agent.plot_frequency_table()
        agent.plot_q_table()
        agent.plot_control()
        #agent.plot_sliced_q_tables()


    # print running avg
    episodes = np.arange(agent.n_episodes)
    for ep in episodes:

        if args.do_report and ep % 1 == 0:
            msg = agent.log_episodes(ep)
            print(msg)

def q_learning(agent, n_steps_lim, alpha):

    # set state space and discretize
    agent.set_state_space()
    agent.discretize_state_space()
    agent.discretize_action_space()

    # initialize q-values table
    agent.initialize_frequency_table()
    agent.initialize_q_table()

    # for each episode
    for ep in np.arange(agent.n_episodes):

        # reset environment
        _ = agent.env.reset()
        state = agent.env.state

        # reset trajectory
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

            # choose action following epsilon greedy action
            idx_action, action = agent.get_epsilon_greedy_action(ep, idx_state)

            # step dynamics forward
            new_obs, r, complete, _ = agent.env.step(action)
            new_state = agent.env.state

            # get idx new state
            idx_new_state = agent.get_state_idx(new_state)

            # get idx state-action pair
            idx = (idx_state, idx_action,)

            # update q values
            agent.n_table[idx] += 1
            agent.q_table[idx] += alpha * (
                  r \
                + agent.gamma * np.max(agent.q_table[(idx_new_state, slice(None))]) \
                - agent.q_table[idx]
            )

            # save reward
            agent.save_reward(r)

            # update state
            state = new_state

        # save q-value
        #agent.q_values = np.concatenate((agent.q_values, q_values[np.newaxis, :]), axis=0)

        # compute return
        agent.compute_discounted_rewards()
        agent.compute_returns()

        # save time steps
        agent.save_episode(ep, k)

        # logs
        if agent.logs:
            msg = agent.log_episodes(ep)
            print(msg)

    # update npz dict
    agent.update_npz_dict_agent()

    # save frequency and q-value last tables
    agent.save_frequency_table()
    agent.save_q_table()

if __name__ == '__main__':
    main()
