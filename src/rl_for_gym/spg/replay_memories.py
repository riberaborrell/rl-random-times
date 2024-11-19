import numpy as np
import torch

from rl_for_gym.utils.replay_memory import ReplayMemory

class ReplayMemoryReturn(ReplayMemory):

    def __init__(self, return_type='n-return', **kwargs):
        super().__init__(**kwargs)

        assert return_type in ['n-return', 'initial-return'], 'Invalid return type'

        # buffer parameters
        self.return_type = return_type

        # initialize arrays and reset counters
        self.reset()

    def reset(self):

        # initialize state, action, n-returns and done arrays 
        self.states = np.full((self.max_size, self.state_dim), np.nan, dtype=np.float32)
        self.actions = np.full((self.max_size, self.action_dim), np.nan, dtype=np.float32)
        if self.return_type == 'n-return':
            self.n_returns = np.full(self.max_size, np.nan, dtype=np.float32)
        else: # self.return_type == 'initial-return':
            self.initial_returns = np.full(self.max_size, np.nan, dtype=np.float32)

        # counters and flags
        self.reset_counters()

    def store(self, state, action, n_return=None, initial_return=None):

        # update buffer arrays
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        if self.return_type == 'n-return':
            self.n_returns[self.ptr] = n_return
        else:
            self.initial_returns[self.ptr] = initial_return

        self.update_store_idx_and_size()


    def store_vectorized(self, states, actions, n_returns=None, initial_returns=None):
        n_experiences = states.shape[0]
        i = self.ptr
        j = self.ptr + n_experiences
        assert j < self.max_size, 'The memory size is too low'

        self.states[i:j] = states
        self.actions[i:j] = actions
        if self.return_type == 'n-return':
            self.n_returns[i:j] = n_returns
        else:
            self.initial_returns[i:j] = initial_returns

        self.ptr = (self.ptr + n_experiences) % self.max_size
        self.size = min(self.size + n_experiences, self.max_size)

    def sample_batch(self, batch_size, replace=True):

        # sample uniformly the batch indexes
        idx = self.sample_batch_idx(batch_size, replace)

        data = dict(
            states=self.states[idx],
            actions=self.actions[idx],
        )
        if self.return_type == 'n-return':
            data['n-returns'] = self.n_returns[idx]
        else:
            data['initial-returns'] = self.initial_returns[idx]

        return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in data.items()}

