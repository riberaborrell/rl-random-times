import numpy as np

def get_idx_done_now(done, already_done):
    idx = np.where(
        (done == True) &
        (already_done == False)
    )[0]
    already_done[idx] = True
    return idx

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

def normalize_advs_trick(x):
    return (x - np.mean(x))/(np.std(x) + 1e-8)


