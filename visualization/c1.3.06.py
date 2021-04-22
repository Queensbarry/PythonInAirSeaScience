import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

x = np.linspace(0.0, 5.0, 100)
y = np.cos(2 * np.pi * x) * np.exp(-x)

def formatoddticks(x, pos):
    "Format odd tick positions"
    if x % 2:
        return '%1.1f' % x
    else:
        return ''

fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
ax.plot(x, y)
formatter = matplotlib.ticker.FuncFormatter(formatoddticks)
locx = matplotlib.ticker.MaxNLocator(nbins=6)
ax.xaxis.set_major_locator(locx)
ax.xaxis.set_major_formatter(formatter)
locy = matplotlib.ticker.MaxNLocator(nbins=5)
ax.yaxis.set_major_locator(locy)

plt.show()
