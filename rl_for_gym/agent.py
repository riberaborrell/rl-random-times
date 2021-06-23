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

        # computation time
        self.t_initial = None
        self.t_final = None

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def set_epsilon_parameters(self, eps_init, eps_min, eps_max, eps_decay):
        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        self.epsilons = [eps_init]

    def update_epsilon_linear_decay(self, eps):
        return self.eps_min + (self.eps_max - self.eps_min) * self.eps_decay * eps

    def update_epsilon_exp_decay(self, episode):
        return self.eps_min + (self.eps_max - self.eps_min) * np.exp( - self.eps_decay * episode)

    def reset_rewards(self):
        self.rewards = np.empty(0)

    def save_reward(self, r):
        self.rewards = np.append(self.rewards, r)

    #TODO! generalize for different types of observation spaces and action spaces
    def reset_trajectory(self, env):
        self.obs_space_dim = env.observation_space.shape[0]

        self.states = np.empty((0, self.obs_space_dim))
        self.actions = np.empty(0, dtype=np.int)
        self.rewards = np.empty(0)

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
        self.total_rewards = np.empty(0)
        self.sample_returns = np.empty(0)
        self.time_steps = np.empty(0, dtype=np.int32)

    def save_episode(self, time_steps):
        self.total_rewards = np.append(self.total_rewards, sum(self.rewards))
        self.sample_returns = np.append(self.sample_returns, self.returns[0])
        self.time_steps = np.append(self.time_steps, time_steps)

    def log_episodes(self, ep, n_avg_episodes=100):
        if ep < n_avg_episodes:
            idx_last_episodes = slice(0, ep)
        else:
            idx_last_episodes = slice(ep - n_avg_episodes, ep)

        msg = 'ep: {:d}, time steps: {:d}, return: {:.2f}, runn avg ({:d}): {:.2f}, ' \
                    'total rewards: {:.2f}, runn avg ({:d}): {:.2f}' \
              ''.format(
                    ep,
                    self.time_steps[ep],
                    self.sample_returns[ep],
                    n_avg_episodes,
                    np.mean(self.sample_returns[idx_last_episodes]),
                    self.total_rewards[ep],
                    n_avg_episodes,
                    np.mean(self.total_rewards[idx_last_episodes])
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
            epsilons=self.epsilons[:-1],
            step_sliced_episodes=self.step_sliced_episodes,
            total_rewards=self.total_rewards,
            sample_returns=self.sample_returns,
            time_steps=self.time_steps,
            q_values=self.q_values,
            t_initial=self.t_initial,
            t_final=self.t_final,
        )

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
            msg = 'no q-learning agent found'
            print(msg)
            return False

