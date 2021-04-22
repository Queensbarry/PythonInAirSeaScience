import matplotlib.pyplot as plt
import numpy as np

# 构造数据
a1 = np.linspace(0, 1.6 * np.pi, 5)
x1 = [np.cos(a) for a in a1]
y1 = [np.sin(a) for a in a1]
a2 = np.linspace(0.2 * np.pi, 1.8 * np.pi, 5)
x2 = [np.cos(a) * 0.4 for a in a2]
y2 = [np.sin(a) * 0.4 for a in a2]
x = np.c_[x1, x2].flatten()
y = np.c_[y1, y2].flatten()

plt.fill(x, y, facecolor='red', edgecolor='orange', linewidth=2)
plt.axis('equal')

plt.show()
