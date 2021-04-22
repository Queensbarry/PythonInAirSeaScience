import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = plt.subplot(211)
ax1.set_ylabel('Water Level')
ax1.set_title('Tide Demo')

x = np.arange(0.0, 1.0, 0.01)
y = np.sin(2 * np.pi * x + 0.35 * np.pi)
line, = ax1.plot(x, y, color = 'k', lw = 2)

ax2 = plt.axes([0.2, 0.1, 0.63, 0.3])
n, bins, patches = ax2.hist(np.random.randn(1000), 50,
                            facecolor = 'm', edgecolor = 'm')
ax2.set_xlabel('Time [sec]')

plt.show()
