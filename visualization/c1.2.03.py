import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
rect = fig.patch
rect.set_facecolor((1.0, 1.0, 0.7))

ax1 = fig.add_axes([0.2, 0.3, 0.4, 0.4])
rect = ax1.patch
rect.set_facecolor((0.3, 0.3, 0.5))

for lbl in ax1.xaxis.get_ticklabels():
    lbl.set_color('red')
    lbl.set_rotation(45)
    lbl.set_fontsize(16)

for line in ax1.yaxis.get_ticklines():
    line.set_color('green')
    line.set_markersize(25)
    line.set_markeredgewidth(3)

plt.show()
