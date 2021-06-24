from agent import QLearningAgent
from base_parser import get_base_parser
from plots import Plot

import gym
from gym.spaces.box import Box

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def normalize_angle(theta):
    ''' returns angle between 0 and 2 pi
    '''
    while (theta < 0 or theta > 2 * np.pi):
        if theta < 0:
            theta += 2 * np.pi
        else:
            theta -= 2 * np.pi
    return theta

def q_learning(agent, n_episodes_lim, n_steps_lim, lr, epsilon, eps_decay, do_render=False):

    # preallocate information for all epochs
    agent.preallocate_episodes()

    # state space
    low = np.array([0, - agent.env.max_speed], dtype=np.float32)
    high = np.array([2 * np.pi, agent.env.max_speed], dtype=np.float32)
    state_space = Box(
        low=low,
        high=high,
        dtype=np.float32,
    )
    state_space_dim = state_space.shape[0]

    # discretize state space
    h_state = np.array([0.05, 0.5])
    state_space_h = np.mgrid[[
        slice(0, 2 * np.pi + h_state[0], h_state[0]),
        slice(-8.0, 8.0 + h_state[1], h_state[1]),
    ]]
    state_space_h = np.moveaxis(state_space_h, 0, -1)

    # discretize action space
    action_space_dim = agent.env.action_space.shape[0]
    h_action = 2
    action_space_h = np.mgrid[[slice(-2.0, 2.0 + h_action, h_action)]]
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
        agent.env.reset()
        state = agent.env.state
        state[0] = normalize_angle(state[0])

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
            idx_state = [None for i in range(state_space_dim)]
            for i in range(state_space_dim):
                axis_i = np.linspace(
                    state_space.low[i],
                    state_space.high[i],
                    state_space_h.shape[i],
                )
                idx_state[i] = np.argmin(np.abs(axis_i - state[i]))

            idx_state = tuple(idx_state)
            #env.state = state_space_h[idx_state]

            # pick greedy action (exploitation)
            if np.random.rand() > epsilon:
                idx_action = np.argmax(q_values[idx_state])
                action = action_space_h[idx_action]

            # pick random action (exploration)
            else:
                action = agent.env.action_space.sample()
                idx_action = np.argmin(np.abs(action_space_h[:, 0] - action))
                #action = action_space_h[idx_action]

            # step dynamics forward
            new_observation, r, complete, _ = agent.env.step(action)
            new_state = agent.env.state
            new_state[0] = normalize_angle(new_state[0])

            #print(k, complete)

            # interpolate new state in our discretized state space
            idx_new_state = [None for i in range(state_space_dim)]
            for i in range(state_space_dim):
                axis_i = np.linspace(
                    state_space.low[i],
                    state_space.high[i],
                    state_space_h.shape[i],
                )
                idx_new_state[i] = np.argmin(np.abs(axis_i - new_state[i]))
            idx_new_state = tuple(idx_new_state)
            #env.state = state_space_h[idx_new_state]

            # update q values
            idx_state_action = idx_state + (idx_action,)
            q_values[idx_state_action] += lr * (
                  #agent.rewards[-1] \
                  r \
                + agent.gamma * np.max(q_values[(idx_new_state[0], idx_new_state[1], slice(None))]) \
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
    env = gym.make('Pendulum-v0')

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


    episodes = np.arange(agent.n_episodes)
    sliced_episodes = episodes[::args.step_sliced_episodes]
    for ep in episodes:

        # print running avg
        if args.do_report and ep % 1 == 0:
            msg = agent.log_episodes(ep)
            print(msg)

        if args.do_plots and ep in sliced_episodes:
            idx_sliced_ep = int(ep / args.step_sliced_episodes)
            v_values = np.max(agent.q_values[idx_sliced_ep], axis=2)

            # print v values table
            #fig, ax = plt.subplots()
            plt.imshow(
                v_values,
                cmap=cm.RdYlGn,
                origin='lower',
                vmax=v_values.max(),
                vmin=v_values.min(),
            )
            plt.colorbar()
            plt.show()



if __name__ == '__main__':
    main()
