import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.linspace(-1,1,150), np.linspace(-1,1,150))
z = (x+y)*np.exp(-5*(x**2+y**2))*6.01-0.152

ax.plot_wireframe(x, y, z, rstride=10, cstride=10)

plt.show()
