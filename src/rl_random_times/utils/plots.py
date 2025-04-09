import matplotlib.pyplot as plt
import numpy as np

from rl_random_times.utils.numeric import compute_running_mean
import rl_random_times.utils.mpl_config

# tableau palettes from matplotlib 
COLORS_TAB10 = [plt.cm.tab10(i) for i in range(20)]
COLORS_TAB20 = [plt.cm.tab20(i) for i in range(20)]
COLORS_TAB20b = [plt.cm.tab20b(i) for i in range(20)]
COLORS_TAB20c = [plt.cm.tab20c(i) for i in range(20)]

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
                 xlabel='', xlim=None, ylim=None, legend=False, loc=None,
                 file_path=None, kwargs_layout={}):

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
    plt.tight_layout(**kwargs_layout)
    plt.savefig(file_path, format='pdf') if file_path is not None else plt.show()

def plot_y_per_episode(x, y, **kwargs):
    plot_y_per_x(x, y, xlabel='Episodes', **kwargs)

def plot_y_per_grad_iteration(x, y, **kwargs):
    plot_y_per_x(x, y, xlabel='Gradient iterations', **kwargs)

def plot_y_per_time_steps(x, y, **kwargs):
    plot_y_per_x(x, y, xlabel='Time steps', **kwargs)

def plot_y_per_ct(x, y, **kwargs):
    plot_y_per_x(x, y, xlabel='Computational time', **kwargs)

def plot_ys_per_x(x, ys, run_window=1, hlines=None, title='', plot_scale='linear',
                  xlabel='', xlim=None, ylim=None, labels=None, colors=None,
                  legend=False, loc=None, file_path=None, kwargs_layout={}):
    n_lines = len(ys)
    if labels is None:
        labels = [None for i in range(n_lines)]
    if colors is None:
        colors = [COLORS_TAB10[i] for i in range(n_lines)]
    if type(x) is not list:
        x = [x for i in range(n_lines)]
    run_mean_ys = [compute_running_mean(y, run_window) if run_window > 1 else None for y in ys]
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    plot_fn = get_plot_function(ax, plot_scale)
    for i in range(n_lines):
        if run_window == 1:
            plot_fn(x[i], ys[i], label=labels[i], color=colors[i], lw=4)
        else:
            plot_fn(x[i], ys[i], color=colors[i], alpha=0.25, lw=4)
            plot_fn(x[i], run_mean_ys[i], label=labels[i], color=colors[i])
    if hlines:
        for (hline, color, ls, label) in hlines:
            ax.axhline(y=hline, c=color, ls=ls, label=label, lw=4.)
    if legend: plt.legend(loc=loc)
    plt.tight_layout(**kwargs_layout)
    plt.savefig(file_path, format='pdf') if file_path is not None else plt.show()

def plot_ys_per_episode(x, ys, **kwargs):
    plot_ys_per_x(x, ys, xlabel='Episodes', **kwargs)

def plot_ys_per_grad_iteration(x, ys, **kwargs):
    plot_ys_per_x(x, ys, xlabel='Gradient iterations', **kwargs)

def plot_ys_per_time_steps(x, ys, **kwargs):
    plot_ys_per_x(x, ys, xlabel='Time steps', **kwargs)

def plot_ys_per_ct(x, ys, **kwargs):
    plot_ys_per_x(x, ys, xlabel='Computational time', **kwargs)

def plot_y_avg_per_x(x, ys, hlines=None, title: str = '', xlabel: str = '', xlim=None, ylim=None,
                     plot_scale='linear', legend: bool = False, loc: str = 'upper right',
                     file_path=None, kwargs_layout={}):
    y = np.mean(ys, axis=0)
    error = np.sqrt(np.var(ys, axis=0))
    fig, ax = plt.subplots()
    plot_fn = get_plot_function(ax, plot_scale)
    ax.set_title(title)#, size=20)
    ax.set_xlabel(xlabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    plot_fn(x, y, label='Mean')
    ax.fill_between(x, y-error, y+error, alpha=0.4, label='Standard deviation')
    if hlines:
        for (hline, color, ls, label) in hlines:
            ax.axhline(y=hline, c=color, ls=ls, label=label)
    if legend: plt.legend(loc=loc)
    plt.tight_layout(**kwargs_layout)
    plt.savefig(file_path, format='pdf') if file_path is not None else plt.show()

def plot_y_avg_per_episode(x, ys, **kwargs):
    plot_y_avg_per_x(x, ys, xlabel='Episodes', **kwargs)

def plot_y_avg_per_grad_iteration(x, ys, **kwargs):
    plot_y_avg_per_x(x, ys, xlabel='Grad. iterations', **kwargs)

def plot_y_avg_per_time_steps(x, ys, **kwargs):
    plot_y_avg_per_x(x, ys, xlabel='Time steps', **kwargs)

def plot_y_avg_per_ct(x, ys, **kwargs):
    plot_y_avg_per_x(x, ys, xlabel='Computational time', **kwargs)

def plot_ys_avg_per_x(x, ys, plot_std=True, hlines=None, title: str = '', xlabel: str = '', xlim=None, ylim=None,
                      plot_scale='linear', labels=None, colors=None, legend: bool = False,
                      loc: str = 'upper right', file_path=None, kwargs_layout={}):
    n_lines = len(ys)
    if labels is None:
        labels = [None for i in range(n_lines)]
    if colors is None:
        colors = [COLORS_TAB10[i] for i in range(n_lines)]
    if type(x) is not list:
        x = [x for i in range(n_lines)]
    ys_mean = [np.mean(y, axis=0) for y in ys]
    errors = [np.sqrt(np.var(y, axis=0)) for y in ys] if plot_std else None
    fig, ax = plt.subplots()
    plot_fn = get_plot_function(ax, plot_scale)
    ax.set_title(title)#, size=20)
    ax.set_xlabel(xlabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    for i in range(n_lines):
        #plot_fn(x[i], ys_mean[i], c=colors[i], label='Mean')
        #ax.fill_between(x[i], ys_mean[i]-errors[i], ys_mean[i]+errors[i], color=colors[i], alpha=0.4, label='Standard deviation') if plot_std else None
        plot_fn(x[i], ys_mean[i], c=colors[i], label=labels[i])
        ax.fill_between(x[i], ys_mean[i]-errors[i], ys_mean[i]+errors[i], color=colors[i], alpha=0.4) if plot_std else None
    if hlines:
        for (hline, color, ls, label) in hlines:
            ax.axhline(y=hline, c=color, ls=ls, label=label)
    if legend: plt.legend(loc=loc)
    plt.tight_layout(**kwargs_layout)
    plt.savefig(file_path, format='pdf') if file_path is not None else plt.show()

def plot_ys_avg_per_episode(x, ys, **kwargs):
    #plot_ys_avg_per_x(x, ys, xlabel='Episodes', **kwargs)
    plot_ys_avg_per_x(x, ys, xlabel='Trajectories', **kwargs)

def plot_ys_avg_per_grad_iteration(x, ys, **kwargs):
    plot_ys_avg_per_x(x, ys, xlabel='Gradient iterations', **kwargs)

def plot_ys_avg_per_time_steps(x, ys, **kwargs):
    plot_ys_avg_per_x(x, ys, xlabel='Time steps', **kwargs)

def plot_ys_avg_per_ct(x, ys, **kwargs):
    plot_ys_avg_per_x(x, ys, xlabel='Computational time', **kwargs)


def plot_lr_grid_search(lrs, ys, title='', plot_scale='loglog', xlim=None, ylim=None, colors=None,
                        labels=None, ls='-', sign=1, file_path=None, kwargs_layout={}):
    n_seeds = ys[0].shape[0]
    fig, ax = plt.subplots()
    plot_fn = get_plot_function(ax, plot_scale)
    ax.set_title(title),
    ax.set_xlabel('Learning rate')
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    for i in range(len(lrs)):
        for j in range(n_seeds):
            if labels is not None:
                plot_fn(lrs[i], sign*ys[i][j], ls=ls, marker='.', ms=15, c=colors[i][j], alpha=0.8, label=labels[i][j])
            else:
                plot_fn(lrs[i], sign*ys[i][j], ls=ls, marker='.', ms=15, c=colors[i][j], alpha=0.8)
    if labels is not None:
        ax.legend()
    plt.tight_layout(**kwargs_layout)
    plt.savefig(file_path, format='pdf') if file_path is not None else plt.show()

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
