import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
import os


PLOT_TYPES = [
    'linear',
    'semilogx',
    'semilogy',
    'loglog',
]

class MyFigure(Figure):
    ''' Figure with one axes.
    '''

    def __init__(self, dir_path, file_name='foo', file_type='png', *args, **kwargs):
        super().__init__(*args, **kwargs)

        # file path attributes
        self.file_name = file_name
        self.file_type = file_type
        self.dir_path = dir_path

        # add set of subplots
        _ = self.subplots()

        # default plot type
        self.plot_type = 'linear'

    @property
    def file_path(self):
        return os.path.join(self.dir_path, self.file_name + '.' + self.file_type)

    def set_font_sizes(self):
        #TODO! revise how **kwargs works

        SMALL_SIZE = 10
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 18

        matplotlib.rc('font', size=SMALL_SIZE)
        matplotlib.rc('axes', titlesize=SMALL_SIZE)
        matplotlib.rc('axes', labelsize=MEDIUM_SIZE)
        matplotlib.rc('xtick', labelsize=SMALL_SIZE)
        matplotlib.rc('ytick', labelsize=SMALL_SIZE)
        matplotlib.rc('legend', fontsize=SMALL_SIZE)
        matplotlib.rc('figure', titlesize=BIGGER_SIZE)

    def set_title(self, title):
        ax = self.axes[0]
        ax.set_title(title)

    def set_xlabel(self, label):
        ax = self.axes[0]
        ax.set_xlabel(label)

    def set_ylabel(self, label):
        ax = self.axes[0]
        ax.set_ylabel(label)

    def set_xlim(self, xmin, xmax):
        ax = self.axes[0]
        ax.set_xlim(xmin, xmax)

    def set_ylim(self, ymin, ymax):
        ax = self.axes[0]
        ax.set_ylim(ymin, ymax)

    def set_plot_type(self, plot_type):
        assert plot_type in PLOT_TYPES, ''
        self.plot_type = plot_type

    def plot_one_line(self, x, y):
        assert x.ndim == y.ndim == 1, ''
        assert x.shape[0] == y.shape[0], ''

        # axes of the figure
        ax = self.axes[0]

        # plot
        if self.plot_type == 'linear':
            ax.plot(x, y)
        elif self.plot_type == 'semilogx':
            ax.semilogx(x, y)
        elif self.plot_type == 'semilogy':
            ax.semilogy(x, y)
        elif self.plot_type == 'loglog':
            ax.loglog(x, y)

        # save figure
        self.savefig(self.file_path)

    def plot_multiple_lines(self, x, y):
        assert x.ndim == 1, ''
        assert y.ndim == 2, ''
        assert x.shape[0] == y.shape[1], ''

        # number of lines to plot
        n_lines = y.shape[0]

        # axes of the figure
        ax = self.axes[0]

        # plot lines
        for i in range(n_lines):
            ax.plot(x, y[i])

        # save figure
        self.savefig(self.file_path)

