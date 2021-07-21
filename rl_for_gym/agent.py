from rl_for_gym.utils_path import get_env_dir_path

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
    def __init__(self, env, gamma=0.99, logs=False):
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

        # greedy
        self.eps_init = None
        self.eps_min = None
        self.eps_max = None
        self.eps_decay = None
        self.epsilons = None

        # batch
        self.batch_states = None
        self.batch_actions = None
        self.batch_total_rewards = None
        self.batch_returns = None
        self.batch_traj_num = None

        # all episodes
        self.n_episodes = None
        self.n_sliced_episodes = None
        self.total_rewards = None
        self.sample_returns = None
        self.time_steps = None
        self.avg_total_rewards = None
        self.avg_sample_returns = None

        # discretized state space and action space
        self.h_state = None
        self.state_space_h = None
        self.h_action = None
        self.action_space_h = None

        # v values and q values tables
        self.v_tables = None
        self.q_tables = None

        # computation time
        self.ct = None

        # save npz file
        self.npz_dict = {}

        # logs flag
        self.logs = logs

    def set_dir_path(self):
        self.dir_path = get_env_dir_path(self.env)

    def start_timer(self):
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        self.ct_final = time.perf_counter()
        self.ct = self.t_final - self.t_initial

    def set_epsilon_parameters(self, eps_init, eps_min, eps_max, eps_decay):
        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        self.epsilons = [eps_init]

    def set_constant_epsilons(self, eps_init):
        self.epsilons = eps_init * np.ones(self.n_episodes)

    def set_glie_epsilons(self):
        self.epsilons = np.array([1 / (ep + 1) for ep in np.arange(self.n_episodes)])

    def set_exp_decay_epsilons(self, eps_min, eps_max, eps_decay):
        self.epsilons = np.array([
            eps_min + (eps_max - eps_min) * np.exp( - eps_decay * ep)
            for ep in np.arange(self.n_episodes)
        ])

    def update_epsilon_exp_decay(self, episode):
        return self.eps_min + (self.eps_max - self.eps_min) * np.exp( - self.eps_decay * episode)

    def update_epsilon_linear_decay(self, eps):
        return self.eps_min + (self.eps_max - self.eps_min) * self.eps_decay * eps


    def reset_rewards(self):
        self.rewards = np.empty(0)

    def save_reward(self, r):
        self.rewards = np.append(self.rewards, r)

    def compute_discounted_rewards(self):
        ''' discount the discount factor to the rewards
        '''
        k_last = self.rewards.shape[0]
        self.discounted_rewards = np.array(
            [self.gamma**k * self.rewards[k] for k in np.arange(k_last)]
        )

    def compute_returns(self):
        ''' computes the obtained return of each state of the trajectory
        '''
        # reverse the discounted reward, apply cumsum and reverse it back
        self.returns = self.discounted_rewards[::-1].cumsum()[::-1]

    def preallocate_episodes(self):
        self.total_rewards = np.empty(self.n_episodes)
        self.avg_total_rewards = np.empty(self.n_episodes)
        self.sample_returns = np.empty(self.n_episodes)
        self.avg_sample_returns = np.empty(self.n_episodes)
        self.time_steps = np.empty(self.n_episodes, dtype=np.int32)

    def save_episode(self, ep, time_steps):

        # get indices episodes to averaged
        if ep < self.n_avg_episodes:
            idx_last_episodes = slice(0, ep + 1)
        else:
            idx_last_episodes = slice(ep + 1 - self.n_avg_episodes, ep + 1)

        self.total_rewards[ep] = sum(self.rewards)
        self.avg_total_rewards[ep] = np.mean(self.total_rewards[idx_last_episodes])
        self.sample_returns[ep] = self.returns[0]
        self.avg_sample_returns[ep] = np.mean(self.sample_returns[idx_last_episodes])
        self.time_steps[ep] = time_steps

    def log_episodes(self, ep):

        msg = 'ep: {:3d}, time steps: {:4d}, return (runn avg ({:d}): {:2.2f}, ' \
                    'total rewards (runn avg ({:d})): {:.2f}' \
              ''.format(
                    ep,
                    self.time_steps[ep],
                    self.n_avg_episodes,
                    self.avg_sample_returns[ep],
                    self.n_avg_episodes,
                    self.avg_total_rewards[ep],
                  )
        return msg


    #TODO! generalize
    def reset_batch(self):
        self.batch_states = np.empty((0, self.obs_space_dim))
        self.batch_actions = np.empty(0, dtype=np.int)
        self.batch_returns = np.empty(0)
        self.batch_traj_num = 0

        self.total_rewards = np.empty(0)


    #TODO! generalize
    def update_batch(self):
        self.batch_states = np.vstack((self.batch_states, self.states))
        self.batch_actions = np.append(self.batch_actions, self.actions)
        self.batch_returns = np.append(self.batch_returns, self.ret)
        self.batch_traj_num += 1

        self.total_rewards = np.append(self.total_rewards, sum(self.rewards))

    def update_npz_dict_agent(self):
        self.npz_dict['n_episodes'] = self.n_episodes
        self.npz_dict['n_avg_episodes'] = self.n_avg_episodes
        self.npz_dict['epsilons'] = self.epsilons
        #step_sliced_episodes=self.step_sliced_episodes
        self.npz_dict['total_rewards'] = self.total_rewards
        self.npz_dict['avg_total_rewards'] = self.avg_total_rewards
        self.npz_dict['sample_returns'] = self.sample_returns
        self.npz_dict['avg_sample_returns'] = self.avg_sample_returns
        self.npz_dict['time_steps'] = self.time_steps
        self.npz_dict['ct'] = self.ct

    def update_npz_dict_last_q_table(self):
        self.npz_dict['last_q_table'] = self.q_tables[-1]

    def update_npz_dict_q_values(self):
        # get sliced episodes
        episodes = np.arange(self.n_episodes)
        self.sliced_episodes = episodes[::self.step_sliced_episodes]
        self.sliced_q_tables = self.q_tables[self.sliced_episodes]

        self.npz_dict['step_sliced_episodes'] = self.step_sliced_episodes
        self.npz_dict['sliced_episodes'] = self.sliced_episodes
        self.npz_dict['sliced_q_values'] = self.sliced_q_tables

    def save(self):
        file_path = os.path.join(self.dir_path, 'agent.npz')
        np.savez(file_path, **self.npz_dict)

    def load(self):
        try:
            file_path = os.path.join(self.dir_path, 'agent.npz')
            data = np.load(file_path, allow_pickle=True)
            for file_name in data.files:
                if data[file_name].ndim == 0:
                    setattr(self, file_name, data[file_name][()])
                else:
                    setattr(self, file_name, data[file_name])
            return True
        except:
            msg = 'no agent found'
            print(msg)
            return False
