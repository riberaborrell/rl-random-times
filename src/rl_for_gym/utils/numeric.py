from typing import Optional

import numpy as np

def compute_running_mean(x: np.array, run_window: Optional[int] = 10):
    ''' running mean / moving average of the array along the given running window.
    '''
    return np.array([
        np.mean(x[i-run_window:i+1]) if i > run_window
        else np.mean(x[:i+1]) for i in range(len(x))
    ])

def compute_running_variance(array: np.array, run_window: Optional[int] = 10):
    ''' running variance of the array along the given running window.
    '''
    return np.array([
        np.var(array[i-run_window:i+1]) if i > run_window
        else np.var(array[:i+1]) for i in range(len(array))
    ])


def cumsum_numpy(x):
    return x[::-1].cumsum()[::-1]

def discount_cumsum(x, gamma):
    n = len(x)
    x = np.array(x)
    y = gamma**np.arange(n)
    z = np.zeros_like(x, dtype=np.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def discount_cumsum2(x, gamma):
    n = len(x)
    y = np.array(
        [gamma**i * x[i] for i in np.arange(n)]
    )
    z = y[::-1].cumsum()[::-1]
    return z
    #return z - z.mean()

def normalize_list(x, eps: Optional[float] = 1e-5):
    ''' Normalize the list by subtracting the mean and dividing by the standard deviation.'''
    return (x - x.mean()) / (x.std() + 1e-5)

def normalize_array(x: np.array, eps: Optional[float] = 1e-5):
    ''' Normalize the array by subtracting the mean and dividing by the standard deviation.'''
    return (x - np.mean(x)) / (np.std(x) + 1e-5)


