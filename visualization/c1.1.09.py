import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

# 水平坐标的空间间隔
dx, dy = 0.05, 0.05

# 产生两个二维坐标网格数组，分别描述坐标x方向和y方向的分布
y, x = np.mgrid[slice(1, 5 + dy, dy), slice(1, 5 + dx, dx)]

# 基于x、y产生一个随空间坐标变化的二维数组
z = np.sin(x)**5 + np.cos(y*x) * np.cos(x)**2

# x和y描述了绘图的范围，z是在这个范围内的某个量的值，
# 对pcolormesh来说，需要将z数组的最后一行（列）去掉
z = z[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

# 选择colormap
cmap = plt.get_cmap('RdYlGn')
#通过levels定义一个基于z中数据的规范化对象实例，它可以把z中的值与levels进行对应转换
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig, (ax0, ax1) = plt.subplots(nrows=2)

im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax0)
ax0.set_title('pcolormesh')


# contours 是基于数据点的绘图方式，所以要将坐标点转换成数据对应的中心点，
# 即对于contour绘图方法来说，坐标点与数据点要对应。
cf = ax1.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1] + dy/2., z,
                  levels=levels, cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('contourf')

# 调整子图之间的距离，使坐标刻度标签和title之间不会重叠
fig.tight_layout()

plt.show()
