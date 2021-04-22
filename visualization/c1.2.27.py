import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms

fig, ax = plt.subplots(figsize=(5, 4))
x, y = 10*np.random.rand(2, 1000)
ax.plot(x, y*10., 'bo', alpha=0.25)
cc = mpatches.Circle((2.5, 2), 1.0, transform=fig.dpi_scale_trans,
                       facecolor='green', alpha=0.75, zorder=3)
ax.add_patch(cc)
plt.show()

fig, ax = plt.subplots(figsize=(7, 3))
x, y = 10*np.random.rand(2, 1000)
ax.plot(x, y*10., 'bo', alpha=0.25)
cc = mpatches.Circle((2.5, 2), 1.0, transform=fig.dpi_scale_trans,
                       facecolor='green', alpha=0.75, zorder=3)
ax.add_patch(cc)
plt.show()
