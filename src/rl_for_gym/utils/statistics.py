import numpy as np

from rl_for_gym.utils.numeric import compute_running_mean
from rl_for_gym.utils.path import load_data, save_data

def compute_array_statistics(array: np.array):
    mean = np.mean(array)
    var = np.var(array)
    std = np.sqrt(var)
    re = std / mean if mean != 0. else np.nan
    return mean, var, std, re

def compute_std_and_re(mean: float, var: float):
    std = np.sqrt(var)
    return std, std / mean


class Statistics(object):

    def __init__(self, eval_freq, eval_batch_size, n_iterations, policy_type='det', iter_str='it.:',
                 track_loss=False, track_ct=False, track_lr=False):

        assert policy_type in ['det', 'stoch', 'stoch-mean'], 'Policy type not recognized'
        self.policy_type = policy_type

        # frequency of evaluation and batch size
        self.eval_freq = eval_freq
        self.eval_batch_size = eval_batch_size

        # number of iterations (episodes, grad. iterations or total steps)
        self.n_iterations = n_iterations
        self.n_epochs = self.n_iterations // eval_freq + 1
        self.iter_str = iter_str

        # flags
        self.track_loss = track_loss
        self.track_ct = track_ct
        self.track_lr = track_lr

        # steps
        self.mean_lengths = np.full(self.n_epochs, np.nan)
        self.var_lengths = np.full(self.n_epochs, np.nan)
        self.max_lengths = np.full(self.n_epochs, np.nan)

        # returns
        self.mean_returns = np.full(self.n_epochs, np.nan)
        self.var_returns = np.full(self.n_epochs, np.nan)

        # losses
        if track_loss:
            self.losses = np.full(self.n_epochs, np.nan)
            self.loss_vars = np.full(self.n_epochs, np.nan)

        # computational time
        if track_ct:
            self.cts = np.full(self.n_epochs, np.nan)

        # learning rates
        if track_lr:
            self.lrs = np.full(self.n_epochs, np.nan)

    def save_epoch(self, i, returns, time_steps, loss=None, loss_var=None, ct=None, lr=None):

        if self.track_loss:
            assert loss is not None and loss_var is not None, 'Loss is not provided'
        if self.track_ct:
            assert ct is not None, 'CT is not provided'
        if self.track_lr:
            assert lr is not None, 'lr is not provided'

        self.mean_lengths[i], self.var_lengths[i], _, _ = compute_array_statistics(time_steps)
        self.max_lengths[i] = np.max(time_steps)
        self.mean_returns[i], self.var_returns[i], _, _ = compute_array_statistics(returns)
        if self.track_loss:
            self.losses[i] = loss
            self.loss_vars[i] = loss_var
        if self.track_ct:
            self.cts[i] = ct
        if self.track_lr:
            self.lrs[i] = lr

    def log_epoch(self, i):
        j = i * self.eval_freq
        msg = self.iter_str + ' {:2d}, '.format(j)
        msg += 'mean return: {:.3e}, var return: {:.1e}, '.format(self.mean_returns[i], self.var_returns[i])
        msg += 'mean lengths: {:.3e}, '.format(self.mean_lengths[i])
        if self.track_loss:
            msg += 'loss: {:.3e}, '.format(self.losses[i])
        if self.track_ct:
            msg += 'ct: {:.3e}, '.format(self.cts[i])
        if self.track_lr:
            msg += 'lr: {:.3e}'.format(self.lrs[i])
        print(msg)

    def save_stats(self, dir_path):
        save_data(self.__dict__, dir_path, file_name='eval-{}.npz'.format(self.policy_type))

    def load_stats(self, dir_path):
        # get data dictionary
        succ, data = load_data(dir_path, file_name='eval-{}.npz'.format(self.policy_type))
        if not succ:
            return

        assert self.eval_freq == data['eval_freq'], 'eval freq mismatch'
        assert self.eval_batch_size == data['eval_batch_size'], 'eval batch size mismatch'
        assert self.n_iterations == data['n_iterations'], 'iterations mismatch'

        # recover attributes
        for key in data:
            setattr(self, key, data[key])

        # compute missing attributes
        self.std_fhts, self.re_fhts = compute_std_and_re(self.mean_fhts, self.var_fhts)

    def get_n_iterations_until_goal(self, attr_name, threshold, sign='smaller', run_window=10):
        assert sign in ['smaller', 'bigger'], 'The inequality sign is not correct'
        if not hasattr(self, attr_name):
            print('The given attribute has not been tracked')
            return np.nan
        run_mean_y = compute_running_mean(getattr(self, attr_name), run_window)
        indices = np.where(run_mean_y > threshold)[0] \
                  if sign == 'bigger' else np.where(run_mean_y < threshold)[0]
        idx = indices[0] * self.eval_freq if len(indices) > 0 else np.nan
        return idx

    def get_stats(self):

        # get number of iterations
        iterations = np.arange(self.n_epochs) * self.eval_freq

        return iterations, self.mean_returns, self.mean_fhts, self.max_lengths

    def get_stats_multiple_datas(self, datas):

        # get number of iterations
        iterations = np.arange(self.n_epochs) * self.eval_freq

        # preallocate arrays
        array_shape = (len(datas), iterations.shape[0])
        objectives = np.empty(array_shape)
        mfhts = np.empty(array_shape)
        max_lengths = np.empty(array_shape)

        # load evaluation
        for i, data in enumerate(datas):
            self.load_stats(data['dir_path'])
            objectives[i] = self.mean_returns
            mfhts[i] = self.mean_fhts
            max_lengths[i] = self.max_lengths

        return iterations, objectives, mfhts, max_lengths
