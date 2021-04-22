from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

# 示例1：不同的zdir方法
zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
xs = (1, 4, 4, 9, 4, 1)
ys = (2, 5, 8, 10, 1, 2)
zs = (10, 3, 8, 9, 1, 8)

for zdir, x, y, z in zip(zdirs, xs, ys, zs):
    label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
    ax.text(x, y, z, label, zdir)

# 示例2：指定颜色
ax.text(9, 0, 0, "Red", color='red')

# 示例3：text2D函数
# 指定位置(0,0)会将文本放置于左下角，而(1,1)则会将文本放置于右上角
ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)

# 调整显示区域和标签
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
