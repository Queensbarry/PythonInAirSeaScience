import matplotlib.pyplot as plt
import numpy as np

Y, X = np.mgrid[-3:3:15j, -3:3:15j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
S = np.sqrt(U**2 + V**2)

fig=plt.figure(figsize=(9,4))

# 用蓝色绘制箭头场
ax1=plt.subplot(121)
q1 = ax1.quiver(X, Y, U, V, color='b')
ax1.quiverkey(q1, X=0.8, Y=1.03, U=10, label='10m/s',labelpos='E')
ax1.set_title('blue quiver')

# 将箭头场的颜色映射到矢量场强度
ax2=plt.subplot(122)
q2 = ax2.quiver(X, Y, U, V, S, cmap=plt.cm.coolwarm)
ax2.quiverkey(q2, X=0.45, Y=-0.1, U=10, label='10m/s',labelpos='E')
ax2.set_title('quiver with colormap')

plt.tight_layout()

plt.show()
