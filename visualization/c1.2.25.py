import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots()
x, y = 10*np.random.rand(2, 1000)
ax.plot(x, y, 'bo', alpha=0.25)
cc = mpatches.Circle((0.5, 0.5), 0.25, transform=ax.transAxes,
                       facecolor='green', alpha=0.75, zorder=3)
ax.add_patch(cc)
plt.show()
