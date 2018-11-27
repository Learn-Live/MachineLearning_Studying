# -*- coding: utf-8 -*-
r"""
    literal plain

    learning seaborn
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_seaborn():
    sns.set_style('darkgrid')

    x = [i for i in range(10)]
    y = [i * np.random.random() for i in range(10)]

    fig, ax = plt.subplots(nrows=1, ncols=2)
    print(ax)
    ax = sns.lineplot(x, y, ax=ax[0])  # if ax is None,  ax = plt.gca(),     p.plot_learning(ax, kwargs)
    # plt.xtricks(rotation=90)
    plt.show()


if __name__ == '__main__':
    plot_seaborn()
