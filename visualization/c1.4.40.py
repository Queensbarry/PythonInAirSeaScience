import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=plt.figaspect(0.5))

x, y, z = get_test_data(0.05)
zlv = np.linspace(-85,85,35)

# 第一个子图
ax1 = fig.add_subplot(1, 2, 1)
ct = ax1.contourf(x, y, z, levels=zlv, cmap=cm.coolwarm)
fig.colorbar(ct, shrink=0.7, aspect=15, ticks=zlv[1::8])

# 第二个子图
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
sf = ax2.plot_surface(x, y, z, rstride=1, cstride=1,
                vmin=zlv[0], vmax=zlv[-1], cmap=cm.bwr)
fig.colorbar(sf, shrink=0.7, aspect=15, orientation='horizontal')

plt.show()
