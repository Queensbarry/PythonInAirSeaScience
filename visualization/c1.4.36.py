from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'y']
yticks = [3, 2, 1, 0]
for c, k in zip(colors, yticks):
    # 为y=k‘层’产生随机数 
    xs = np.arange(20)
    ys = np.random.rand(20)

    cs = [c] * len(xs)
    cs[0] = 'c'

    # 用xs和ys在y=k平面上以80%的透明度绘制条形图
    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_yticks(yticks)

plt.show()
