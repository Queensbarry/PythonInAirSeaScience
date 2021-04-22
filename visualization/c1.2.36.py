import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

cdict = {'red':   [[0.0,  0.0, 0.0],
                   [0.5,  1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'green': [[0.0,  0.0, 0.0],
                   [0.25, 0.0, 0.0],
                   [0.75, 1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.5,  0.0, 0.0],
                   [1.0,  1.0, 1.0]]}

cdict['red'] = [[0.0,  0.0, 0.3],
                [0.5,  1.0, 0.6],
                [1.0,  1.0, 1.0]]

def plot_linearmap(cdict):
    newcmp = LinearSegmentedColormap('testCmap',
                                     segmentdata=cdict, N=256)
    rgba = newcmp(np.linspace(0, 1, 256))
    fig, ax = plt.subplots(figsize=(4, 3),
                            constrained_layout=True)
    col = ['r', 'g', 'b']
    xs = [0, 0.25, 0.5, 0.75, 1]
    for xx in xs:
        ax.axvline(xx, color='0.7', linestyle='--')
    for i in range(3):
        ax.plot(np.arange(256)/256, rgba[:, i], color=col[i], lw=3)
    ax.set_xticks(xs)
    ax.set_xlabel('Index')
    ax.set_ylabel('RGB')
    plt.show()

plot_linearmap(cdict)

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cmap = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
cb.set_label('Discontinued LinearSegmentedColormap')
fig.show()
