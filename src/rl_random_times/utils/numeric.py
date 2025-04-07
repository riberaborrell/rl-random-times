from typing import Optional, Union

import numpy as np
import scipy
import torch

# vectorized operations
def dot_vect(a, b):
    return (a * b).sum(axis=1)

def dot_vect_torch(a, b):
    return torch.matmul(
        torch.unsqueeze(a, dim=1),
        torch.unsqueeze(b, dim=2),
    ).squeeze()


def compute_running_mean(x: np.array, run_window: Optional[int] = 10) -> np.array:
    ''' running mean / moving average of the array along the given running window.
    '''
    return np.array([
        np.mean(x[i-run_window:i+1]) if i > run_window
        else np.mean(x[:i+1]) for i in range(len(x))
    ])

def compute_running_variance(array: np.array, run_window: Optional[int] = 10) -> np.array:
    ''' running variance of the array along the given running window.
    '''
    return np.array([
        np.var(array[i-run_window:i+1]) if i > run_window
        else np.var(array[:i+1]) for i in range(len(array))
    ])


def cumsum_numpy(x):
    return x[::-1].cumsum()[::-1]

def discounted_cumsum_list(x: list, gamma: float) -> list:
    discounted_cumsum = []
    discounted_sum = 0
    for el in reversed(x):
        discounted_sum = el + gamma * discounted_sum
        discounted_cumsum.insert(0, discounted_sum)
    return discounted_cumsum

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

def normalize_array(x: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-5):
    ''' Normalize the np.array or torch.tensor by subtracting the mean
        and dividing by the standard deviation.
    '''
    assert x.ndim == 1, 'Input must be a 1D array'
    return (x - x.mean()) / (x.std() + eps)

def interpolate_array(xs, ys, n_points=100):

    assert xs.ndim == ys.ndim == 2, ''

    common_x = np.linspace(xs.min(axis=1).max(), xs.max(axis=1).min(), n_points)

    # interpolate y-array onto the common x grid
    interpolated_y = []
    for x, y in zip(xs, ys):
        interp_func = scipy.interpolate.interp1d(
            x, y, kind='linear', bounds_error=False, fill_value="extrapolate",
        )
        interpolated_y.append(interp_func(common_x))

    return common_x, np.array(interpolated_y)
