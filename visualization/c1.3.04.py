import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

x = np.linspace(0.0, 5.0, 100)
y = np.cos(2 * np.pi * x) * np.exp(-x)

fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
axs[0].plot(x, y)
axs[1].plot(x, y)
axs[1].xaxis.set_ticks(np.arange(0., 8.1, 2.))
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
axs[0].plot(x, y)
axs[1].plot(x, y)
ticks = np.arange(0., 8.1, 2.)
ticklbl = ['%1.2f' % tick for tick in ticks]
axs[1].xaxis.set_ticks(ticks)
axs[1].xaxis.set_ticklabels(ticklbl)
axs[1].set_xlim(axs[0].get_xlim())
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
axs[0].plot(x, y)
axs[1].plot(x, y)
ticks = np.arange(0., 8.1, 2.)
formatter = matplotlib.ticker.StrMethodFormatter('{x:1.1f}')
axs[1].xaxis.set_ticks(ticks)
axs[1].xaxis.set_major_formatter(formatter)
axs[1].set_xlim(axs[0].get_xlim())
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
axs[0].plot(x, y)
axs[1].plot(x, y)
ticks = np.arange(0., 8.1, 2.)
formatter = matplotlib.ticker.StrMethodFormatter('{x:1.1f}')
locator = matplotlib.ticker.FixedLocator(ticks)
axs[1].xaxis.set_major_locator(locator)
axs[1].xaxis.set_major_formatter(formatter)
plt.show()


