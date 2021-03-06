import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict
cmaps = OrderedDict()

cmaps['Perceptually Uniform Sequential'] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']

cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

cmaps['Sequential (2)'] = [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']

cmaps['Diverging'] = [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']

cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']

cmaps['Miscellaneous'] = [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

cmaps_items = [i for i in cmaps.items()]
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

def plot_color_gradients(cmap_category, cmap_list, nrows):
    fig, axes = plt.subplots(figsize=(6.4, 0.3*nrows), nrows=nrows)
    fig.subplots_adjust(top=1 - 0.27/0.3/nrows,
                        bottom=0.054/0.3/nrows,
                        left=0.2, right=0.99)
    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    for ax in axes:
        ax.set_axis_off()

# Sequential
for n in range(3):
    cmap_category, cmap_list = cmaps_items[n]
    nrows = len(cmap_list)
    plot_color_gradients(cmap_category, cmap_list, nrows)
plt.show()

# Diverging
n = 3
cmap_category, cmap_list = cmaps_items[n]
nrows = len(cmap_list)
plot_color_gradients(cmap_category, cmap_list, nrows)
plt.show()

# Cyclic
n = 4
cmap_category, cmap_list = cmaps_items[n]
nrows = len(cmap_list)
plot_color_gradients(cmap_category, cmap_list, nrows)
plt.show()

# Qualitative
n = 5
cmap_category, cmap_list = cmaps_items[n]
nrows = len(cmap_list)
plot_color_gradients(cmap_category, cmap_list, nrows)
plt.show()

# Miscellaneous
n = 6
cmap_category, cmap_list = cmaps_items[n]
nrows = len(cmap_list)
plot_color_gradients(cmap_category, cmap_list, nrows)
plt.show()

