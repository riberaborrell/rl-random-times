from agent import Agent
from gym.spaces.box import Box

from figures import MyFigure
from matplotlib import cm

import numpy as np

class SdeAgent(Agent):
    '''
    '''

    def set_state_space(self):
        ''' define state space
        '''
        self.state_space = Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            shape=(1,),
            dtype=np.float32,
        )

    def discretize_state_space(self):
        ''' discretize state space
        '''
        self.h_state = 0.5
        state_space_h = np.mgrid[[
            slice(self.state_space.low[0], self.state_space.high[0] + self.h_state, self.h_state),
        ]]
        self.state_space_h = np.moveaxis(state_space_h, 0, -1)

    def discretize_action_space(self):
        ''' discretize action space
        '''
        action_space = self.env.action_space
        self.h_action = 0.5
        action_space_h = np.mgrid[[
            slice(action_space.low[0], action_space.high[0] + self.h_action, self.h_action)
        ]]
        self.action_space_h = np.moveaxis(action_space_h, 0, -1)

    def get_state_idx(self, state):
        ''' interpolate state in our discretized state space and get corresponding index
        '''
        idx_state = np.argmin(np.abs(self.state_space_h[:, 0] - state))
        return idx_state

    def get_action_idx(self, action):
        '''
        '''
        idx_action = np.argmin(np.abs(self.action_space_h[:, 0] - action))
        return idx_action

    def reset_trajectory(self):
        self.states = np.empty(0)
        self.actions = np.empty(0)
        self.rewards = np.empty(0)

    def save_history(self, state, action, r):
        self.states = np.append(self.states, state)
        self.actions = np.append(self.actions, action)
        self.rewards = np.append(self.rewards, r)

    def preallocate_tables(self):
        self.q_tables = np.empty(
            (0,) + self.state_space_h.shape[:-1] + action_space_h.shape[:-1]
        )

    def initialize_frequency_table(self):
        self.n_table = np.zeros(
            self.state_space_h.shape[:-1] + self.action_space_h.shape[:-1]
        )

    def initialize_q_table(self):
        self.q_table = np.random.rand(*
            self.state_space_h.shape[:-1] + self.action_space_h.shape[:-1]
        )

    def initialize_eligibility_traces(self):
        self.e_table = np.zeros(
            self.state_space_h.shape[:-1] + self.action_space_h.shape[:-1]
        )

    def save_frequency_table(self):
        self.last_n_table = self.n_table
        self.npz_dict['last_n_table'] = self.n_table

    def save_q_table(self):
        self.last_q_table = self.q_table
        self.npz_dict['last_q_table'] = self.q_table

    def get_epsilon_greedy_action(self, ep, idx_state):
        # get epsilon
        epsilon = self.epsilons[ep]

        # pick greedy action (exploitation)
        if np.random.rand() > epsilon:
            idx_action = np.argmax(self.q_table[idx_state])
            action = self.action_space_h[idx_action]

        # pick random action (exploration)
        else:
            action = self.env.action_space.sample()
            idx_action = self.get_action_idx(action)

        return idx_action, action



    def plot_total_rewards(self):
        fig = MyFigure(self.dir_path, 'total_rewards')
        y = np.vstack((self.total_rewards, self.avg_total_rewards))
        fig.set_ylim(-100, 0)
        fig.plot_multiple_lines(self.episodes, y)

    def plot_time_steps(self):
        fig = MyFigure(self.dir_path, 'time_steps')
        fig.plot_one_line(self.episodes, self.time_steps)

    def plot_epsilons(self):
        fig = MyFigure(self.dir_path, 'epsilons')
        fig.set_plot_type('semilogy')
        fig.plot_one_line(self.episodes, self.epsilons)

    def plot_frequency_table(self):
        # set extent bounds
        left = self.state_space_h[0, 0]
        right = self.state_space_h[-1, 0]
        bottom = self.action_space_h[0, 0]
        top = self.action_space_h[-1, 0]

        fig = MyFigure(self.dir_path, 'frequency_table')
        im = fig.axes[0].imshow(
            self.last_n_table.T,
            origin='lower',
            extent=(left, right, bottom, top),
            cmap=cm.coolwarm,
        )

        # add space for colour bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        # save figure
        fig.savefig(fig.file_path)

    def plot_q_table(self):

        # set extent bounds
        left = self.state_space_h[0, 0]
        right = self.state_space_h[-1, 0]
        bottom = self.action_space_h[0, 0]
        top = self.action_space_h[-1, 0]

        # see https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html

        fig = MyFigure(self.dir_path, 'q_table')
        im = fig.axes[0].imshow(
            self.last_q_table.T,
            origin='lower',
            extent=(left, right, bottom, top),
            cmap=cm.viridis,
        )

        # add space for colour bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        # save figure
        fig.savefig(fig.file_path)

    def plot_control(self):
        x = self.state_space_h[:, 0]
        control = np.empty_like(x)

        for idx, x_k in enumerate(x):
            idx_action = np.argmax(self.last_q_table[idx])
            control[idx] = self.action_space_h[idx_action]

        fig = MyFigure(self.dir_path, 'control')
        fig.plot_one_line(x, control)

    def plot_sliced_q_tables(self):
        for idx, ep in enumerate(self.sliced_episodes):
            q_values = self.sliced_q_values[idx]
            q_values = np.moveaxis(q_values, 0, -1)
            plt.imshow(
                q_values,
                cmap=cm.RdYlGn,
                origin='lower',
                vmin=q_values.min(),
                vmax=q_values.max(),
                extent=[-3, 3, 0, 5],
            )
            plt.colorbar()
            plt.show()
