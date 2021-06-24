import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import os

class Plot:
    def __init__(self, dir_path, file_name=None, file_type='png'):
        self.file_name = file_name
        self.file_type = file_type
        self.dir_path = dir_path

        # title and label
        self.title = ''
        self.label = ''

        # axes labels
        self.xlabel = None
        self.ylabel = None
        self.zlabel = None

        # axes bounds
        self.xmin = None
        self.xmax= None
        self.ymin = None
        self.ymax= None
        self.zmin = None
        self.zmax = None

        # axes ticks
        self.yticks = None

    @property
    def file_path(self):
        if self.file_name:
            return os.path.join(
                self.dir_path, self.file_name + '.' + self.file_type
            )
        else:
            return None

    def set_title(self, title):
        self.plt.title(title)

    def set_label(self, label):
        self.label = label

    def set_xlim(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    def set_ylim(self, ymin, ymax):
        self.ymin = ymin
        self.ymax = ymax

    def set_zlim(self, zmin, zmax):
        self.zmin = zmin
        self.zmax = zmax

    def one_line_plot(self, n_episodes, y):
        x = np.arange(n_episodes)
        plt.plot(x, y)
        plt.xlabel('Episodes')
        plt.savefig(self.file_path)
        plt.close()
