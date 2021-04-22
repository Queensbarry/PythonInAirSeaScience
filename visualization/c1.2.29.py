import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms

fig, ax = plt.subplots()

# 绘制正统曲线
x = np.linspace(0., 2*np.pi, 100)
y = np.sin(2*x)
line, = ax.plot(x, y, lw=5, color='blue')

# 通过曲线对象向右3点并向下3点创建偏移变换
dx, dy = 3/72., -3/72.
offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
shadow_transform = ax.transData + offset

# 用偏移变换绘制相同的数据，并调整zorder确保其被绘制于曲线之下
ax.plot(x, y, lw=3, color='gray',
        transform=shadow_transform,
        zorder=0.5*line.get_zorder())

ax.set_title('creating a shadow effect with an offset transform')
plt.show()
