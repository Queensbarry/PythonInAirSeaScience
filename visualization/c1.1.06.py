import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(np.pi*t)
plt.plot(t, s, lw=2)

plt.annotate('Local Maximum', xy=(2, 1), xytext=(3, 1.5), color='r',
             arrowprops=dict(facecolor='r', edgecolor='r', shrink=0.05))

plt.ylim(-2, 2)
plt.show()
