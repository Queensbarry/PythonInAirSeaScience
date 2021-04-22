import matplotlib.pyplot as plt
import numpy as np

r = np.random.rand(24) * 2.8  + 0.2
theta = np.arange(0,2 * np.pi, np.pi / 12.)

ax = plt.subplot(111, projection='polar')
ax.bar(theta, r, width=np.pi/9., color='g', alpha=0.7)
ax.set_rmax(3)
ax.set_rticks(np.arange(0.5, 3.1, 0.5))

ax.grid(True)

ax.set_title("A bar plot on a polar axis", va='bottom')
plt.show()
