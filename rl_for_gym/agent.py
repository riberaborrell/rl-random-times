from rl_for_gym.utils_path import get_q_learning_agent_dir_path

import numpy as np
import time

from pathlib import Path
import os

SOURCE_PATH = Path(os.path.dirname(__file__))
PROJECT_PATH = SOURCE_PATH.parent
DATA_PATH = os.path.join(PROJECT_PATH, 'data')

class Agent:
    '''
    '''
    def __init__(self, env, gamma=0.99):
        assert 0 <= gamma <= 1, ''

        # environment information
        self.env = env
        self.obs_space_dim = None
        self.state_space_dim = None
        self.action_space_dim = None

        # episode
        self.states = None
        self.actions = None
        self.rewards = None

        self.gamma = gamma
        self.discounted_rewards = None
        self.returns = None

        # batch
        self.batch_states = None
        self.batch_actions = None
        self.batch_total_rewards = None
        self.batch_returns = None
        self.batch_traj_num = None

        # all episodes
        self.n_episodes = None
        self.total_rewards = None
        self.all_returns = None
        self.all_time_steps = None

        # computation time
        self.t_initial = None
        self.t_final = None

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def preallocate_episodes(self):

        # all episodes
        self.total_rewards = np.empty(0)
        self.all_returns = np.empty(0)
        self.all_time_steps = np.empty(0, dtype=np.int32)

    def save_episode(self, time_steps):
        self.total_rewards = np.append(self.total_rewards, sum(self.rewards))
        self.all_returns = np.append(self.all_returns, self.returns[0])
        self.all_time_steps = np.append(self.all_time_steps, time_steps)

    def reset_rewards(self):
        self.rewards = np.empty(0)

    def save_reward(self, r):
        self.rewards = np.append(self.rewards, r)

    def reset_trajectory(self, env):
        self.obs_space_dim = env.observation_space.shape[0]

        self.states = np.empty((0, self.obs_space_dim))
        self.actions = np.empty(0, dtype=np.int)
        self.rewards = np.empty(0)

    def compute_discounted_rewards(self):
        k_last = self.rewards.shape[0]
        self.discounted_rewards = np.array(
            [self.gamma**k * self.rewards[k] for k in np.arange(k_last)]
        )

    def compute_returns(self):
        '''computes the obtained return of each state of the trajectory. Also saves
           the initial return for each trajectory
        '''
        # reverse the discounted reward, apply cumsum and reverse it back
        self.returns = self.discounted_rewards[::-1].cumsum()[::-1]

    def get_v_value_table(self):
        pass


    def reset_batch(self):
        self.batch_states = np.empty((0, self.obs_space_dim))
        self.batch_actions = np.empty(0, dtype=np.int)
        self.batch_returns = np.empty(0)
        self.batch_traj_num = 0

        self.total_rewards = np.empty(0)


    def update_batch(self):
        self.batch_states = np.vstack((self.batch_states, self.states))
        self.batch_actions = np.append(self.batch_actions, self.actions)
        self.batch_returns = np.append(self.batch_returns, self.ret)
        self.batch_traj_num += 1

        self.total_rewards = np.append(self.total_rewards, sum(self.rewards))


class QLearningAgent(Agent):
    '''
    '''

    def __init__(self, env, gamma=0.99):
        '''
        '''
        super().__init__(env, gamma)

        # discretized state space and action space
        self.h_state = None
        self.state_space_h = None
        self.h_action = None
        self.action_space_h = None

        # v values and q values tables
        self.v_values = None
        self.q_values = None

    def set_dir_path(self):
        self.dir_path = get_q_learning_agent_dir_path(self.env)

    def preallocate_tables(self, state_space_h, action_space_h):
        self.state_space_h = state_space_h
        self.action_space_h = action_space_h

        self.q_values = np.empty((0,) + state_space_h.shape[:-1] + action_space_h.shape[:-1])
        #self.q_values = np.zeros(state_space_h.shape[:-1] + action_space_h.shape[:-1])

    def save(self):
        file_path = os.path.join(self.dir_path, 'agent.npz')
        np.savez(
            file_path,
            n_episodes=self.n_episodes,
            total_rewards=self.total_rewards,
            all_returns=self.all_returns,
            all_time_steps=self.all_time_steps,
            v_values=self.v_values,
            q_values=self.q_values,
            t_initial=self.t_initial,
            t_final=self.t_final,
        )

    def load(self):
        try:
            agent = np.load(
                os.path.join(self.dir_path, 'agent.npz'),
                allow_pickle=True,
            )
            self.n_episodes = agent['n_episodes']
            self.total_rewards = agent['total_rewards']
            self.all_returns = agent['all_returns']
            self.all_time_steps = agent['all_time_steps']
            self.v_values = agent['v_values']
            self.q_values = agent['q_values']
            self.t_initial = agent['t_initial']
            self.t_final = agent['t_final']
            return True

        except:
            msg = 'no q-learning agent found'
            print(msg)
            return False

