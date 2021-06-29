from agent import QLearningAgent
from gym.spaces.box import Box

from figures import MyFigure

import numpy as np

class SdeAgent(QLearningAgent):
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
        self.h_state = 0.01
        state_space_h = np.mgrid[[
            slice(self.state_space.low[0], self.state_space.high[0] + self.h_state, self.h_state),
        ]]
        self.state_space_h = np.moveaxis(state_space_h, 0, -1)

    def discretize_action_space(self):
        ''' discretize action space
        '''
        action_space = self.env.action_space
        self.h_action = 0.1
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

    def plot_q_table_control(self):
        x = self.state_space_h[:, 0]
        control = np.empty_like(x)

        for idx, x_k in enumerate(x):
            idx_action = np.argmax(self.last_q_values[idx])
            control[idx] = self.action_space_h[idx_action]

        # plot q-table control
        fig = MyFigure(self.dir_path, 'control')
        fig.plot_one_line(x, control)

