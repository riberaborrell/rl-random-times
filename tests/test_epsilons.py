from rl_for_gym.agent import Agent
from rl_for_gym.figures import MyFigure

import numpy as np
import gym
import pytest

class TestEpsilons:
    '''
    '''

    @pytest.fixture
    def agent(self, n_episodes):
        ''' agent
        '''
        # get env
        env = gym.make('Blackjack-v0')

        # initialize agent object
        agent = Agent(env, gamma=0.99)

        # set number of episodes
        agent.n_episodes = n_episodes

        return agent

    def test_glie_epsilons(self, agent):
        agent.set_glie_epsilons()

        # plot epsilons
        x = np.arange(agent.n_episodes)
        y = agent.epsilons
        fig = MyFigure('data/tests/epsilons', 'glie')
        fig.set_plot_type('semilogy')
        fig.plot_one_line(x, y)

    def test_exp_decay_epsilons(self, agent):
        agent.set_exp_decay_epsilons(eps_min=0.1, eps_max=1, eps_decay=0.01)

        # plot epsilons
        x = np.arange(agent.n_episodes)
        y = agent.epsilons
        fig = MyFigure('data/tests/epsilons', 'exp_decay')
        fig.set_plot_type('semilogy')
        fig.plot_one_line(x, y)

    def test_mc_epsilons(self, n_episodes):
        eps_init = 1
        eps_decay = 0.999
        eps_min = 0.9
        epsilons = np.empty(n_episodes)
        epsilons[0] = eps_init

        for ep in np.arange(1, n_episodes):
            epsilons[ep] = np.maximum(epsilons[ep-1] * eps_decay, eps_min)

        x = np.arange(n_episodes)
        y = epsilons
        fig = MyFigure('data/tests/epsilons', 'blackjack')
        fig.set_plot_type('semilogy')
        fig.plot_one_line(x, y)
