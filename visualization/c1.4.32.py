from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# 构造数据
n_r, n_a, pi2 = 8, 36, 2*np.pi
radii = np.linspace(0.125, 1.0, n_r)
angles = np.linspace(0, pi2, n_a, endpoint=False)[..., np.newaxis]
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())
z = np.sin(-x*y)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, z, linewidth=1, cmap=cm.cool)

plt.show()
