import numpy as np
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

ln1, = plt.plot([1, 2.5, 3], 'r-*', ms=11)
ln2, = plt.plot([3, 1.5, 1], 'b-o', ms=8)

lgd = plt.legend([(ln1, ln2)], ['Two-keys'],
                numpoints=1, handlelength=4,
                handler_map={tuple: HandlerTuple(ndivide=None)})

plt.show()
