import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.random.rand(50)+0.1, 'rx', ms=10)
plt.plot(np.random.rand(50)+0.4, 'r*', ms=10)
plt.plot(np.random.rand(50)+0.7, 'bx', ms=10)
plt.plot(np.random.rand(50)+0.9, 'b*', ms=10)

red_patch = mpatches.Patch(color='red', label='The red data')
blue_patch = mpatches.Patch(color='blue', label='The blue data')
plt.legend(handles=[red_patch, blue_patch])

plt.show()
