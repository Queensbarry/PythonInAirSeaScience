import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

ln1, = plt.plot([3, 2.5, 1], marker = 'o', label = 'line 1')
ln2, = plt.plot([1, 1.5, 3], marker = 'o', label = 'line 2')
plt.legend(handler_map = {ln1: HandlerLine2D(numpoints = 3),
                          ln2: HandlerLine2D(numpoints = 2)})
plt.show()
