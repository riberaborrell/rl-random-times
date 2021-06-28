from agent import QLearningAgent
from base_parser import get_base_parser
from plots import Plot

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

    # define state space
    state_space = Box(
        low=agent.env.observation_space.low[0],
        high=agent.env.observation_space.high[0],
        shape=(1,),
        dtype=np.float32,
    )

    # discretize state space
    h_state = 0.1
    state_space_h = np.mgrid[[
        slice(state_space.low[0], state_space.high[0] + h_state, h_state),
    ]]
    state_space_h = np.moveaxis(state_space_h, 0, -1)

    # discretize action space
    action_space = agent.env.action_space
    h_action = 0.1
    action_space_h = np.mgrid[[
        slice(action_space.low[0], action_space.high[0] + h_action, h_action)
    ]]
    action_space_h = np.moveaxis(action_space_h, 0, -1)

    #
    agent.preallocate_tables(state_space_h, action_space_h)

    # initialize q-values table
    q_values = np.zeros(state_space_h.shape[:-1] + action_space_h.shape[:-1])

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

            # interpolate state in our discretized state space
            idx_state = np.argmin(np.abs(state_space_h[:, 0] - state))

            # pick greedy action (exploitation)
            if np.random.rand() > epsilon:
                idx_action = np.argmax(q_values[idx_state])
                action = action_space_h[idx_action]

            # pick random action (exploration)
            else:
                action = agent.env.action_space.sample()
                idx_action = np.argmin(np.abs(action_space_h[:, 0] - action))

            # step dynamics forward
            new_obs, r, complete, _ = agent.env.step(action)
            new_state = agent.env.state

            #print(k, complete)

            # interpolate new state in our discretized state space
            idx_new_state = np.argmin(np.abs(state_space_h[:, 0] - new_state))

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
        epsilon = agent.update_epsilon_linear_decay(epsilon)
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
    agent = QLearningAgent(env, args.gamma)

    # get dir path
    agent.set_dir_path('q-learning')

    # run q-learning agent
    if not args.load:

        # set epsilon
        agent.set_epsilon_parameters(args.epsilon, args.eps_min, args.eps_max, args.eps_decay)

        # q-learning
        q_learning(agent, args.n_episodes_lim, args.n_steps_lim,
                   args.lr, args.epsilon, args.eps_decay)

        # save agent
        agent.step_sliced_episodes = args.step_sliced_episodes
        agent.save()
        return

    # load already run agent
    else:
        if not agent.load():
            return

    # plot total rewards and epsilons
    if args.do_plots:

        # plot sample returns
        plot = Plot(agent.dir_path, 'sample_returns')
        plot.one_line_plot(agent.n_episodes, agent.sample_returns)

        # plot total rewards
        plot = Plot(agent.dir_path, 'total_rewards')
        plot.one_line_plot(agent.n_episodes, agent.total_rewards)

        # plot time steps
        plot = Plot(agent.dir_path, 'time_steps')
        plot.one_line_plot(agent.n_episodes, agent.time_steps)

        # plot epsilons
        plot = Plot(agent.dir_path, 'epsilons')
        plot.one_line_plot(agent.n_episodes, agent.epsilons)
        return


    episodes = np.arange(agent.n_episodes)
    sliced_episodes = episodes[::args.step_sliced_episodes]
    for ep in episodes:

        # print running avg
        if args.do_report and ep % 1 == 0:
            msg = agent.log_episodes(ep)
            print(msg)

        if args.do_plots and ep in sliced_episodes:
            idx_sliced_ep = int(ep / args.step_sliced_episodes)
            #v_values = np.max(agent.q_values[idx_sliced_ep], axis=1)
            q_values = agent.q_values[idx_sliced_ep]

            # print v values table
            #fig, ax = plt.subplots()
            plt.imshow(
                q_values,
                cmap=cm.RdYlGn,
                origin='lower',
                vmin=q_values.min(),
                vmax=q_values.max(),
            )
            plt.colorbar()
            plt.show()



if __name__ == '__main__':
    main()
