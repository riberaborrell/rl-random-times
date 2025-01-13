import numpy as np

def simple_lr_schedule(it, lr_init=1e-5, lr_final=1e-3, n_iter=1000):
    """
    Compute the learning rate at a given iteration using a logarithmic schedule.

    Parameters:
    - it: current iteration
    - lr_init: initial learning rate.
    - lr_final: final learning rate.
    - n_iter: total number of iterations.

    Returns:
    - Learning rate at the given iteration.
    """

    it = min(it, n_iter-1)
    final_log_scale_factor = np.log10(lr_final) - np.log10(lr_init)

    #scale_factor = 0 + (it - 1) * (final_log_scale_factor / (n_iter - 1))
    #return lr_init * 10 ** scale_factor
    return np.logspace(0, final_log_scale_factor, n_iter)[1:][it-1]

def two_phase_lr_schedule(it, lr_init=1e-5, lr_final=1e-3, n_iter_adaptive=1000):
    """
    Compute the learning rate at a given iteration.

    Parameters:
    - it: current iteration.
    - lr_init: initial learning rate.
    - lr_final: final learning rate.
    - n_iter_adaptive: number of iterations during the increasing or decreasing phase

    Returns:
    - Learning rate at the given iteration.
    """


    final_log_scale_factor = np.log10(lr_final) - np.log10(lr_init)

    # increasing or decreasing phase (from lr_init to lr_final)
    if it < n_iter_adaptive:
        return np.logspace(0, final_log_scale_factor, n_iter_adaptive)[1:][it-1]

    # constant phase
    else:
        return 10**final_log_scale_factor

