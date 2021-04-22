import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 构造数据
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
S = np.sqrt(U**2 + V**2)

fig = plt.figure(figsize=(7, 9))
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

# 沿流线设置变化的流线密度
ax0 = fig.add_subplot(gs[0, 0])
ax0.streamplot(X, Y, U, V, density=[0.5, 1], color='b')
ax0.set_title('Varying Density')

# 沿流线设置变化的颜色
ax1 = fig.add_subplot(gs[0, 1])
strm = ax1.streamplot(X, Y, U, V, color=U, linewidth=2, cmap='plasma')
fig.colorbar(strm.lines)
ax1.set_title('Varying Color')

# 沿流线设置变化的线条宽度
ax2 = fig.add_subplot(gs[1, 0])
lw = 5 * S / S.max()
ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)
ax2.set_title('Varying Line Width')

# 控制流线的起点Controlling the starting points of the streamlines
seed_points = np.array([[-2, -1, 0, 1, 2, -1], [-2, -1,  0, 1, 2, 2]])
ax3 = fig.add_subplot(gs[1, 1])
strm = ax3.streamplot(X, Y, U, V, color=S, linewidth=2,
                     cmap='plasma', start_points=seed_points.T)
fig.colorbar(strm.lines)
ax3.set_title('Controlling Starting Points')

# 用蓝色符号标记流线起点
ax3.plot(seed_points[0], seed_points[1], 'bo')
ax3.set(xlim=(-3, 3), ylim=(-3, 3))

# 创建一个mask
mask = np.zeros(U.shape, dtype=bool)
mask[40:60, 40:60] = True
U[:20, :20] = np.nan
U = np.ma.array(U, mask=mask)

ax4 = fig.add_subplot(gs[2:, :])
ax4.streamplot(X, Y, U, V, color='g')
ax4.set_title('Streamplot with Masking')

ax4.imshow(~mask, extent=(-3, 3, -3, 3), alpha=0.5,
          interpolation='nearest', cmap='summer', aspect='auto')
ax4.set_aspect('equal')

plt.tight_layout()
plt.show()
