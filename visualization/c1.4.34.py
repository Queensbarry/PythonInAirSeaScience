from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
x,y = np.meshgrid(np.linspace(-1,1,150), np.linspace(-1,1,150))
z = (x+y)*np.exp(-5*(x**2+y**2))*6.01-0.152

cset = ax.contourf(x, y, z, 13, cmap=cm.jet)

ax.clabel(cset, fontsize=9, inline=1)

plt.show()

