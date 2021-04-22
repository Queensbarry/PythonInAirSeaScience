from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

# 在x-y轴平面绘制一个sin曲线
x = np.linspace(0, 1, 100)
y = np.sin(x * 2 * np.pi) / 2 + 0.5
ax.plot(x, y, color='r', zs=0, zdir='z', label='curve in (x,y)')

# 创建条形图数据
x = np.linspace(0.1, 0.9, 9)
y = np.sin(x*np.pi)
# 通过设置zdir='y'，并设置zs=0，将条形图绘制于x-z轴平面
ax.bar(x, y, color='b', width=0.08, zs=0, zdir='y',
             label='patchs in (x,z), zs=0', alpha=0.7)
# 创建随机散点数据
x = np.random.sample(100)
y = np.random.sample(100)
# 通过设置zdir='y'，并设置zs=1，将这些点绘制于x-z轴平面
ax.scatter(x, y, s=15, zs=1, zdir='y', c='g', marker='s',
             label='points in (x,z), zs=1')

# 绘制图例，并且设置各坐标轴的范围和标签
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置3D坐标轴的视角，以便更容易观察散点图与sin曲线的位置
ax.view_init(elev=20., azim=-25)

plt.show()
