from sde_agent import SdeAgent
from base_parser import get_base_parser
from figures import MyFigure

import gym
import gym_sde
from gym.spaces.box import Box

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def q_learning(agent, n_episodes_lim, n_steps_lim, lr, epsilon, eps_decay, do_render=False):

    # preallocate information for all epochs
    agent.preallocate_episodes()

    # set state space and discretize
    agent.set_state_space()
    agent.discretize_state_space()
    agent.discretize_action_space()

    agent.preallocate_tables(agent.state_space_h, agent.action_space_h)

    # initialize q-values table
    q_values = np.zeros(agent.state_space_h.shape[:-1] + agent.action_space_h.shape[:-1])

    # set epsilon
    epsilon = agent.eps_init

    # different trajectories
    for ep in np.arange(n_episodes_lim):

        # reset environment
        _ = agent.env.reset()
        state = agent.env.state

        # reset trajectory
        agent.reset_rewards()

        # assign 0 as first reward
        #agent.save_reward(0)

        # terminal state flag
        complete = False

        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            if do_render:
                agent.env.render()

            # get index of the state
            idx_state = agent.get_state_idx(state)

            # pick greedy action (exploitation)
            if np.random.rand() > epsilon:
                idx_action = np.argmax(q_values[idx_state])
                action = agent.action_space_h[idx_action]

            # pick random action (exploration)
            else:
                action = agent.env.action_space.sample()

            # step dynamics forward
            new_obs, r, complete, _ = agent.env.step(action)
            new_state = agent.env.state

            #print(k, complete)

            # get idx new state
            idx_new_state = agent.get_state_idx(new_state)

            # update q values
            idx_state_action = (idx_state,) + (idx_action,)
            q_values[idx_state_action] += lr * (
                  r \
                + agent.gamma * np.max(q_values[(idx_new_state, slice(None))]) \
                - q_values[idx_state_action]
            )

            # save reward
            agent.save_reward(r)

            # update state
            state = new_state


        # save q-value
        agent.q_values = np.concatenate((agent.q_values, q_values[np.newaxis, :]), axis=0)

        # compute return
        agent.compute_discounted_rewards()
        agent.compute_returns()

        # save time steps
        agent.save_episode(time_steps=k)

        # update epsilon
        #epsilon = agent.update_epsilon_linear_decay(epsilon)
        agent.epsilons.append(epsilon)

        # logs
        msg = agent.log_episodes(ep)
        print(msg)
        #print('{:.0}'.format(agent.total_rewards[-1:]))

    # save number of episodes
    agent.n_episodes = ep + 1


def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('sde-v0')

    # initialize Agent
    agent = SdeAgent(env, args.gamma)

    # get dir path
    agent.set_dir_path('q-learning')

    # run q-learning agent
    if not args.load:

        # set epsilon
        agent.set_epsilon_parameters(args.eps_init, args.eps_min, args.eps_max, args.eps_decay)

        # q-learning
        q_learning(agent, args.n_episodes_lim, args.n_steps_lim,
                   args.lr, args.eps_init, args.eps_decay)

        # save agent
        agent.step_sliced_episodes = args.step_sliced_episodes

        # save npz file
        agent.update_npz_dict_agent()
        agent.update_npz_dict_q_values()
        agent.save()

    # load already run agent
    else:
        if not agent.load():
            return

        # set state space and discretize
        agent.set_state_space()
        agent.discretize_state_space()
        agent.discretize_action_space()

    # plot total rewards and epsilons
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

        # plot epsilons
        fig = MyFigure(agent.dir_path, 'epsilons')
        fig.plot_one_line(episodes, agent.epsilons[:-1])

        # plot q-table control
        agent.plot_q_table_control()
        return

        # plot q-table 
        for idx, ep in enumerate(agent.sliced_episodes):
            q_values = agent.sliced_q_values[idx]
            plt.imshow(
                q_values,
                cmap=cm.RdYlGn,
                origin='lower',
                vmin=q_values.min(),
                vmax=q_values.max(),
            )
            plt.colorbar()
            plt.show()


    # print running avg
    episodes = np.arange(agent.n_episodes)
    for ep in episodes:

        if args.do_report and ep % 1 == 0:
            msg = agent.log_episodes(ep)
            print(msg)


if __name__ == '__main__':
    main()
