import numpy as np

from rl_for_gym.spg.reinforce_stochastic_core import ReinforceStochastic
from rl_for_gym.utils.numeric import compute_running_mean

def get_coarse_lrs(lr_low, lr_high):
    assert lr_low < lr_high, ''
    assert np.log10(lr_low) % 1 == 0, ''
    assert np.log10(lr_high) % 1 == 0, ''

    e_low, e_high = int(np.log10(lr_low)), int(np.log10(lr_high))
    lrs = []
    for e in range(e_low, e_high+1):
        lrs.append(10**(e))
    return lrs

def get_fine_lrs(lr_low, lr_high):
    assert lr_low < lr_high, ''
    assert np.log10(lr_low) % 1 == 0, ''
    assert np.log10(lr_high) % 1 == 0, ''

    e_low, e_high = int(np.log10(lr_low)), int(np.log10(lr_high))
    lrs = []
    for e in range(e_low, e_high):
        lrs.append(10**(e))
        lrs.append(2*10**(e))
        lrs.append(5*10**(e))
    lrs.append(10**(e_high))
    return lrs

def get_n_iterations_until_goal(data, threshold, run_window=100):
    ''' get the number of iterations until the mean return is bigger than the threshold'''
    run_mean_y = compute_running_mean(data['mean_returns'], run_window)
    indices = np.where(run_mean_y > threshold)[0]
    idx = indices[0] if len(indices) > 0 else np.nan
    return idx

def get_z_factor_experiment(env, kwargs, kwargs_rt, kwargs_op, lrs, seeds, threshold, run_window):

    # check if lrs is a list of 3 lists
    assert isinstance(lrs, list), "The object is not a list."
    assert len(lrs) == 3, "The list does not have 3 elements."
    for i, item in enumerate(lrs):
        assert isinstance(item, list), f"The element at index {i} is not a list."

    # check if seeds is a list
    assert isinstance(seeds, list), "The object is not a list."

    def get_info(data, threshold, run_window):

        if 'mean_returns' not in data.keys():
            return np.nan, np.nan, np.nan, np.nan

        # get the last not nan values of the mean_returns
        not_nan_mask = ~np.isnan(data['mean_returns'])
        indices = np.where(not_nan_mask)[0]
        last_idx = indices[-1]
        last_key_values = data['mean_returns'][last_idx-run_window:last_idx].mean()

        # get the number of iterations until the threshold is reached
        n_iter = get_n_iterations_until_goal(data, threshold, run_window)

        # get the total number of time steps done at each iteration 
        time_steps = data['max_lengths'][slice(0, n_iter)].sum() if n_iter is not np.nan else np.nan

        # get the computational time at each iteration 
        cts = data['cts'][slice(0, n_iter)].sum() if n_iter is not np.nan else np.nan

        return last_key_values, n_iter, time_steps, cts

    # preallocate arrays
    lasts = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]
    n_grad_iters = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]
    time_steps = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]
    cts = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]

    for i, seed in enumerate(seeds):

        # random time horizon
        for j, lr in enumerate(lrs[0]):
            agent = ReinforceStochastic(env, lr=lr, seed=seed, **kwargs, **kwargs_rt)
            succ, data = agent.run_reinforce(load=True)
            if succ:
                lasts[0][i, j], n_grad_iters[0][i, j], time_steps[0][i, j], cts[0][i, j] \
                    = get_info(data, threshold, run_window)

        # on policy expectation with z-factor estimated 
        for j, lr in enumerate(lrs[1]):
            agent = ReinforceStochastic(env, estimate_z=True, lr=lr, seed=seed, **kwargs, **kwargs_op)
            succ, data = agent.run_reinforce(load=True)
            if succ:
                lasts[1][i, j], n_grad_iters[1][i, j], time_steps[1][i, j], cts[1][i, j] \
                    = get_info(data, threshold, run_window)

        # on policy expectation with z-factor neglected 
        for j, lr in enumerate(lrs[2]):
            agent = ReinforceStochastic(env, estimate_z=False, lr=lr, seed=seed, **kwargs, **kwargs_op)
            succ, data = agent.run_reinforce(load=True)
            if succ:
                lasts[2][i, j], n_grad_iters[2][i, j], time_steps[2][i, j], cts[2][i, j] \
                    = get_info(data, threshold, run_window)

    return lasts, n_grad_iters, time_steps, cts

