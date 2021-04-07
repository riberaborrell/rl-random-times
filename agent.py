import numpy as np

class Agent:
    '''
    '''
    def __init__(self, env, gamma=0.99):
        assert 0 <= gamma <= 1, ''

        self.obs_space_dim = env.observation_space.shape[0]

        self.states = None
        self.actions = None
        self.rewards = None
        self.gamma = gamma
        self.discounted_rewards = None
        self.ret = None
        self.total_rewards = np.empty(0)

        self.batch_states = None
        self.batch_actions = None
        self.batch_returns = None
        self.batch_traj_num = None

    def reset_trajectory(self):
        self.states = np.empty((0, self.obs_space_dim))
        self.actions = np.empty(0, dtype=np.int)
        self.rewards = np.empty(0)

    def get_discounted_rewards_and_returns(self):
        k_last = self.rewards.shape[0]
        self.discounted_rewards = np.array(
            [self.gamma**k * self.rewards[k] for k in np.arange(k_last)]
        )
        # reverse the discounted reward, apply cumsum and reverse it back
        self.ret = self.discounted_rewards[::-1].cumsum()[::-1]

    def reset_batch(self):
        self.batch_states = np.empty((0, self.obs_space_dim))
        self.batch_actions = np.empty(0, dtype=np.int)
        self.batch_returns = np.empty(0)
        self.batch_traj_num = 0

    def update_batch(self):
        self.batch_states = np.vstack((self.batch_states, self.states))
        self.batch_actions = np.append(self.batch_actions, self.actions)
        self.batch_returns = np.append(self.batch_returns, self.ret)
        self.batch_traj_num += 1

        self.total_rewards = np.append(self.total_rewards, sum(self.rewards))
