import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
fig = plt.figure()
l1 = lines.Line2D([0, 1], [0, 1], color = 'red',
                    linewidth = 3, linestyle = '--',
                    transform = fig.transFigure, figure = fig)
l2 = lines.Line2D([0, 1], [1, 0], color = 'blue',
                    linewidth = 2, linestyle = ':',
                    transform = fig.transFigure, figure = fig)
fig.add_artist(l1)
fig.add_artist(l2)
plt.show()
