import numpy as np

def compute_running_mean(array, run_window=10):
    ''' computes the running mean / moving average of the given array along the given running window.
    '''
    return np.array([
        np.mean(array[i-run_window:i+1]) if i > run_window
        else np.mean(array[:i+1]) for i in range(len(array))
    ])

def compute_running_variance(array, run_window=10):
    ''' computes the running variance of the given array along the given running window.
    '''
    return np.array([
        np.var(array[i-run_window:i+1]) if i > run_window
        else np.var(array[:i+1]) for i in range(len(array))
    ])

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

def normalize_advs_trick(x, eps=1e-8):
    ''' normalize reward
    '''
    #return (x - np.mean(x))/(np.std(x) + eps)
    return (x - x.mean()) / (x.std() + eps)


