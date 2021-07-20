from sde_agent import SdeAgent
from base_parser import get_base_parser
from utils_path import get_sarsa_lambda_dir_path

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
    agent.set_dir_path('sarsa-lambda')
    agent.dir_path = get_sarsa_lambda_dir_path(agent.dir_path, 'glie', args.alpha,
                                               args.lam, args.n_episodes_lim)

    # run sarsa lambda agent
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
        sarsa_lambda(agent, args.n_steps_lim, args.alpha, args.lam)

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

def sarsa_lambda(agent, n_steps_lim, alpha, lam):

    # set state space and discretize
    agent.set_state_space()
    agent.discretize_state_space()
    agent.discretize_action_space()

    # initialize q-values table
    agent.initialize_frequency_table()
    agent.initialize_q_table()
    agent.initialize_eligibility_traces()

    # for each episode
    for ep in np.arange(agent.n_episodes):

        # reset environment and choose action
        _ = agent.env.reset()
        state = agent.env.state
        idx_state = agent.get_state_idx(state)
        idx_action, action = agent.get_epsilon_greedy_action(ep, idx_state)

        # reset rewards
        agent.reset_rewards()

        # terminal state flag
        complete = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # step dynamics forward
            new_obs, r, complete, _ = agent.env.step(action)
            new_state = agent.env.state
            idx_new_state = agent.get_state_idx(new_state)

            # get new action
            idx_new_action, new_action = agent.get_epsilon_greedy_action(ep, idx_new_state)


            # get idx state-action pair
            idx = (idx_state, idx_action,)
            idx_new = (idx_new_state, idx_new_action,)

            # update frequency table
            agent.n_table[idx] += 1

            # compute temporal difference error
            td_error = r + agent.gamma * agent.q_table[idx_new] - agent.q_table[idx]

            # update eligibility traces table
            agent.e_table[idx] += 1

            # update the whole q-value and eligibility traces tables
            agent.q_table = agent.q_table + alpha * td_error * agent.e_table
            agent.e_table = agent.e_table * agent.gamma * lam

            # save reward
            agent.save_reward(r)

            # update state and action
            state = new_state
            action = new_action

        # compute return
        agent.compute_discounted_rewards()
        agent.compute_returns()

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
