import numpy as np

from rl_for_gym.spg.reinforce_stochastic_core import ReinforceStochastic
from rl_for_gym.dpg.reinforce_deterministic_core import ReinforceDeterministic
from rl_for_gym.utils.numeric import compute_running_mean

LEARNING_RATES = [
    1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6,
    1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2,
    1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0,
    1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3,
]

def get_coarse_lrs(lr_low, lr_high):
    assert LEARNING_RATES[0] <= lr_low < lr_high <= LEARNING_RATES[-1], ''
    return [
        lr for lr in LEARNING_RATES
            if lr >= lr_low and lr <= lr_high and np.log10(lr) % 1 == 0
    ]

def get_fine_lrs(lr_low, lr_high):
    assert LEARNING_RATES[0] <= lr_low < lr_high <= LEARNING_RATES[-1], ''
    return [lr for lr in LEARNING_RATES if lr >= lr_low and lr <= lr_high]


def get_n_iterations_until_goal(data, threshold, run_window=100):
    ''' get the number of iterations until the mean return is bigger than the threshold'''
    run_mean_y = compute_running_mean(data['mean_returns'], run_window)
    indices = np.where(run_mean_y > threshold)[0]
    idx = indices[0] if len(indices) > 0 else np.nan
    return idx

def get_z_factor_experiment(env, kwargs, kwargs_rt, kwargs_op, kwargs_op_unbiased,
                            kwargs_op_biased, lrs, seeds, threshold, run_window=100):

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

        # get the number of time steps done by the longest trajectory at each iteration
        time_steps = data['max_lengths'][slice(0, n_iter)].sum() if n_iter is not np.nan else np.nan

        # get the total number of time steps done at each iteration
        #total_steps = data['total_lengths'][slice(0, n_iter)].sum() if n_iter is not np.nan else np.nan

        # get the computational time at each iteration 
        cts = data['cts'][slice(0, n_iter)].sum() if n_iter is not np.nan else np.nan

        return last_key_values, n_iter, time_steps, cts
        #return last_key_values, n_iter, total_steps, cts

    # deterministic policy or stochastic policy
    ReinforceClass = ReinforceStochastic if 'policy_type' in kwargs else ReinforceDeterministic

    # preallocate arrays
    lasts = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]
    n_grad_iters = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]
    time_steps = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]
    total_steps = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]
    cts = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]

    for i, seed in enumerate(seeds):

        # set seed
        kwargs['seed'] = seed

        # random time horizon
        for j, lr in enumerate(lrs[0]):
            agent = ReinforceClass(env, lr=lr, **kwargs, **kwargs_rt)
            succ, data = agent.run_reinforce(load=True)
            if succ:
                lasts[0][i, j], n_grad_iters[0][i, j], time_steps[0][i, j], cts[0][i, j] \
                    = get_info(data, threshold, run_window)

        # on policy expectation with z-factor estimated 
        for j, lr in enumerate(lrs[1]):
            agent = ReinforceClass(env, lr=lr, **kwargs, **kwargs_op, **kwargs_op_unbiased)
            succ, data = agent.run_reinforce(load=True)
            if succ:
                lasts[1][i, j], n_grad_iters[1][i, j], time_steps[1][i, j], cts[1][i, j] \
                    = get_info(data, threshold, run_window)

        # on policy expectation with z-factor neglected 
        for j, lr in enumerate(lrs[2]):
            agent = ReinforceClass(env, lr=lr, **kwargs, **kwargs_op, **kwargs_op_biased)
            succ, data = agent.run_reinforce(load=True)
            if succ:
                lasts[2][i, j], n_grad_iters[2][i, j], time_steps[2][i, j], cts[2][i, j] \
                    = get_info(data, threshold, run_window)

    return lasts, n_grad_iters, time_steps, cts

def find_optimal_lr(lrs, lasts, maximize=True):
    idx, optimal_lrs, max_lasts_avg = [], [], []
    for i in range(3):
        lasts_avg = np.mean(lasts[i], axis=0)
        if maximize:
            idx.append(np.nanargmax(lasts_avg))
        else:
            idx.append(np.nanargmin(lasts_avg))
        max_lasts_avg.append(lasts_avg[idx[i]])
        optimal_lrs.append(lrs[i][idx[i]])
    return idx, optimal_lrs, max_lasts_avg
