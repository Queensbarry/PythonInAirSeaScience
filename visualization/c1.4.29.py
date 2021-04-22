from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs =  20 + 12 * np.random.rand(100)
ys =  50 + 35 * np.random.rand(100)
zs = -20 + 50 * np.random.rand(100)
ax.scatter(xs, ys, zs, s=64, c='g', marker='^')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
