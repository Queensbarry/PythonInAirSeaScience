import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
pink = np.array([248/256, 24/256, 148/256, 1])
newcolors[:50, :] = pink
newcmp = ListedColormap(newcolors)


def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3),
                            constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap,
                            rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

plot_examples([viridis, newcmp])

viridis512 = cm.get_cmap('viridis', 512)
newcmp = ListedColormap(viridis512(np.linspace(0.25, 0.75, 256)))
plot_examples([viridis, newcmp])

bottom = cm.get_cmap('Blues_r', 128)
top = cm.get_cmap('Reds', 128)

newcolors = np.vstack((bottom(np.linspace(0, 1, 128)),
                       top(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='BlueRed')
plot_examples([viridis, newcmp])

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(96/256, 1, N)
vals[:, 1] = np.linspace(32/256, 1, N)
vals[:, 2] = np.linspace(160/256, 1, N)
newcmp = ListedColormap(vals)
plot_examples([viridis, newcmp])
