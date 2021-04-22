import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from itertools import product


def twist_xy(a, b, c, d):
    i=np.linspace(0.0, 2*np.pi, 72)
    return np.sin(i*a)+np.cos(i*b), np.sin(i*c)+np.cos(i*d)


fig = plt.figure(figsize=(8, 8))

outer_grid = fig.add_gridspec(4, 4, wspace=0.0, hspace=0.0)

for i in range(16):
    inner_grid = outer_grid[i].subgridspec(3, 3, wspace=0.0, hspace=0.0)
    a, b = int(i / 4) + 1, i % 4 + 1
    for j, (c, d) in enumerate(product(range(1, 4), repeat=2)):
        ax = fig.add_subplot(inner_grid[j])
        ax.plot(*twist_xy(a, b, c, d))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

all_axes = fig.get_axes()

for ax in all_axes:
    for sp in ax.spines.values():
        sp.set_visible(False)
    if ax.is_first_row():
        ax.spines['top'].set_visible(True)
    if ax.is_last_row():
        ax.spines['bottom'].set_visible(True)
    if ax.is_first_col():
        ax.spines['left'].set_visible(True)
    if ax.is_last_col():
        ax.spines['right'].set_visible(True)

plt.show()
