from figures import MyFigure

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import os

SOURCE_PATH = Path(os.path.dirname(__file__))
PROJECT_PATH = SOURCE_PATH.parent
DATA_PATH = os.path.join(PROJECT_PATH, 'data')

def main():

    # create data
    x = np.linspace(-3, 3, 201)
    y = np.tanh(x) + 0.1 * np.cos(5 * x)

    # instantate MyFigure object
    fig = MyFigure(dir_path=DATA_PATH, file_name='bar')

    # set title
    fig.set_title('bla')
    fig.set_xlabel('x')
    fig.set_ylabel('y')

    # plot
    fig.plot_one_line(x, y)



if __name__ == '__main__':
    main()
