from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

def polygon_under_graph(xlist, ylist):
    # 创建填充(xlist, ylist)线图下方空间的多边形顶点列表，假设xs是升序的
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]

fig = plt.figure()
ax = fig.gca(projection='3d')

verts = []
xs = np.linspace(0., 10., 26)

# 第i个多边形会被放置于y=zs[i]平面
zs = range(4)

for i in zs:
    ys = np.random.rand(len(xs))
    verts.append(polygon_under_graph(xs, ys))

poly = PolyCollection(verts,
        facecolors=['r', 'g', 'b', 'y'], alpha=.6)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, 10)
ax.set_ylim(-1, 4)
ax.set_zlim(0, 1)

plt.show()
