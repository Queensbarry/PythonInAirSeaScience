import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch

styles = mpatch.BoxStyle.get_styles()
spacing = 1.2

figheight = spacing * np.floor(len(styles) / 2 + 1) * 1.1
fig = plt.figure(figsize=(5.4, figheight / 1.5))
fontsize = 0.3 * 72

for i, stylename in enumerate(sorted(styles)):
    xs = 0.25 + i % 2 * 0.5
    ys = 1 - (np.floor(i / 2 + 1) - 0.2) * spacing / figheight
    fig.text(xs, ys,
            stylename, ha="center", va="center",
            size=fontsize, transform=fig.transFigure,
            bbox=dict(boxstyle=stylename, fc="w", ec="k"))

plt.show()
