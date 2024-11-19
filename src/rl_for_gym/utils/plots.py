import matplotlib.pyplot as plt
import numpy as np

from rl_for_gym.utils.numeric import compute_running_mean

def get_plot_function(ax, plot_scale):
    if plot_scale == 'linear':
        return ax.plot
    elif plot_scale == 'semilogx':
        return ax.semilogx
    elif plot_scale == 'semilogy':
        return ax.semilogy
    elif plot_scale == 'loglog':
        return ax.loglog
    else:
        raise ValueError('plot_scale must be one of: lineal, semilogx, semilogy, loglog')

def plot_y_per_x(x, y, run_window=1, hlines=None, title='', plot_scale='linear',
                 xlabel='', xlim=None, ylim=None, legend=False, loc=None, file_path=None):

    run_mean_y = compute_running_mean(y, run_window) if run_window > 1 else None
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    plot_fn = get_plot_function(ax, plot_scale)
    plot_fn(x, y, c='tab:blue', alpha=0.5)
    if run_window > 1:
        plot_fn(x, run_mean_y, label='running mean', c='tab:blue')
    if hlines:
        for (hline, color, ls, label) in hlines:
            ax.axhline(y=hline, c=color, ls=ls, label=label)
    if legend: plt.legend(loc=loc)
    plt.savefig(file_path, format='pdf') if file_path is not None else plt.show()

def plot_y_per_episode(x, y, **kwargs):
    plot_y_per_x(x, y, xlabel='Episodes', **kwargs)

def plot_y_per_grad_iteration(x, y, **kwargs):
    plot_y_per_x(x, y, xlabel='Grad. iterations.', **kwargs)

def plot_y_per_time_steps(x, y, **kwargs):
    plot_y_per_x(x, y, xlabel='Time steps', **kwargs)


def plot_fht_histogram(time_steps, n_steps_lim=None):

    if n_steps_lim is None:
        n_steps_lim = time_steps.max()

    x = np.arange(n_steps_lim)
    counts, bins = np.histogram(time_steps, bins=x, density=True)
    fig, ax = plt.subplots()
    ax.hist(bins[:-1], bins, weights=counts, alpha=0.5, label=r'histogram')
    ax.legend()
    plt.show()

def plot_y_episodes(y, smooth=True, ylabel=''):
    run_mean_y = compute_running_mean(y, 10)
    plt.figure(figsize=(12, 8))
    plt.plot(y)
    plt.plot(run_mean_y)
    plt.ylabel(ylabel)
    plt.xlabel('Episodes')
    plt.show()
