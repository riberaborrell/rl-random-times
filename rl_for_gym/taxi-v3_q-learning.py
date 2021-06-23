from agent import QLearningAgent
from base_parser import get_base_parser
from plots import Plot

import gym
from gym.spaces.box import Box

import numpy as np
import matplotlib.pyplot as plt


def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def q_learning(agent, n_episodes_lim, n_steps_lim, lr, do_render=False):

    # preallocate information for all epochs
    agent.preallocate_episodes()

    # preallocate q-value table
    agent.q_values = np.empty((0, agent.env.observation_space.n, agent.env.action_space.n))

    # initialize q-values table
    q_values = np.zeros((agent.env.observation_space.n, agent.env.action_space.n))

    # set epsilon
    epsilon = agent.eps_init

    # different trajectories
    for ep in np.arange(n_episodes_lim):

        # reset environment
        state = agent.env.reset()

        # reset trajectory
        agent.reset_rewards()

        # terminal state flag
        complete = False

        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            if do_render:
                agent.env.render()

            # pick greedy action (exploitation)
            if np.random.rand() > epsilon:
                action = np.argmax(q_values[state])

            # pick random action (exploration)
            else:
                action = agent.env.action_space.sample()

            # step dynamics forward
            new_state, r, complete, _ = agent.env.step(action)

            # update q values
            q_values[state, action] += lr * (
                  r \
                + agent.gamma * np.max(q_values[new_state, :]) \
                - q_values[state, action]
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
        epsilon = agent.update_epsilon_exp_decay(ep)
        agent.epsilons.append(epsilon)


    # save number of episodes
    agent.n_episodes = ep + 1


def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('Taxi-v3')

    # initialize Agent
    agent = QLearningAgent(env, args.gamma)

    # get dir path
    agent.set_dir_path()

    # run q-learning agent
    if not args.load:

        # set epsilon
        agent.set_epsilon_parameters(args.epsilon, args.eps_min, args.eps_max, args.eps_decay)

        # q-learning
        q_learning(agent, args.n_episodes_lim, args.n_steps_lim, args.lr)

        # save agent
        agent.step_sliced_episodes = args.step_sliced_episodes
        agent.save()
        return

    # load already run agent
    else:
        if not agent.load():
            return

    # do plots
    if args.do_plots:
        plt = Plot(agent.dir_path, 'total_rewards')
        plt.plot_total_rewards(agent.n_episodes, agent.total_rewards)
        plt = Plot(agent.dir_path, 'epsilons')
        plt.plot_total_rewards(agent.n_episodes, agent.epsilons)


    if args.do_report:
        for ep in np.arange(agent.n_episodes):

            # print running avg
            if ep % 1 == 0:
                msg = agent.log_episodes(ep)
                print(msg)


if __name__ == '__main__':
    main()
