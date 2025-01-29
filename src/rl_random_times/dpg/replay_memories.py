import numpy as np
import torch

from rl_random_times.utils.replay_memory import ReplayMemory


class ReplayMemoryModelBasedDPG(ReplayMemory):
    ''' replay memory for storing state, dbts and n-returns or initial returns.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # initialize arrays and reset counters
        self.reset()

    def reset(self):

        # initialize arrays
        self.states = np.full((self.max_size, self.state_dim), np.nan, dtype=np.float32)
        self.dbts = np.full((self.max_size, self.action_dim), np.nan, dtype=np.float32)
        self.returns = np.full(self.max_size, np.nan, dtype=np.float32)

        # counters and flags
        self.reset_counters()

    def store(self, state, dbt, ret):

        # update memory
        self.states[self.ptr] = state
        self.dbts[self.ptr] = dbt
        self.returns[self.ptr] = ret
        self.update_store_idx_and_size()

    def store_vectorized(self, states, dbts, returns):
        n_experiences = states.shape[0]
        i = self.ptr
        j = self.ptr + n_experiences
        if j > self.max_size:
            raise ValueError('Replay Memory is too small!')

        # update buffer
        self.states[i:j] = states
        self.dbts[i:j] = dbts
        self.returns[i:j] = returns

        self.ptr = (self.ptr + n_experiences) % self.max_size
        self.size = min(self.size + n_experiences, self.max_size)

    def sample_batch(self, batch_size, replace=True):

        # sample uniformly the batch indexes
        idx = self.sample_batch_idx(batch_size, replace)

        data = dict(
            states=torch.as_tensor(self.states[idx], dtype=torch.float32),
            dbts=torch.as_tensor(self.dbts[idx], dtype=torch.float32),
            returns=torch.as_tensor(self.returns[idx], dtype=torch.float32),
        )
        return data
