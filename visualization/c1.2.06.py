import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

gcline = mlines.Line2D([], [], color='green', marker='x',
                          markersize=12, label='Green Cross')
plt.legend(handles=[gcline])

plt.show()
