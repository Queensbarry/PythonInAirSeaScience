import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def get_demo_image():
    import numpy as np
    x,y = np.meshgrid(np.linspace(-1,1,15),
                      np.linspace(-1,1,15))
    z = (x+y)*np.exp(-5*(x**2+y**2))*6.01-0.152
    # z is a numpy array of 15x15
    return z, (-3, 4, -4, 3)

fig, (ax, ax2) = plt.subplots(ncols=2, figsize=[6, 3])

# 第一个子图
ax.set_aspect(1)
axins = zoomed_inset_axes(ax, zoom=0.5, loc='upper right')
# 修改Inset Axes的刻度值
axins.yaxis.get_major_locator().set_params(nbins=7)
axins.xaxis.get_major_locator().set_params(nbins=7)

plt.setp(axins.get_xticklabels(), visible=False)
plt.setp(axins.get_yticklabels(), visible=False)

def add_sizebar(ax, size):
    asb = AnchoredSizeBar(ax.transData,
                          size,
                          str(size),
                          loc=8,
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)

add_sizebar(ax, 0.5)
add_sizebar(axins, 0.5)

# 第二个子图
Z, extent = get_demo_image()
Z2 = np.zeros([150, 150], dtype="d")
ny, nx = Z.shape
Z2[30:30 + ny, 30:30 + nx] = Z
# extent = [-3, 4, -4, 3]
ax2.imshow(Z2, extent=extent, interpolation="nearest",
          origin="lower")

axins2 = zoomed_inset_axes(ax2, 6, loc=1)  # zoom = 6
axins2.imshow(Z2, extent=extent, interpolation="nearest",
              origin="lower")

# 原图像的子区域
x1, x2, y1, y2 = -1.6, -0.9, -2.6, -1.9
axins2.set_xlim(x1, x2)
axins2.set_ylim(y1, y2)
# 修改Inset Axes的刻度值
axins2.yaxis.get_major_locator().set_params(nbins=7)
axins2.xaxis.get_major_locator().set_params(nbins=7)

plt.setp(axins2.get_xticklabels(), visible=False)
plt.setp(axins2.get_yticklabels(), visible=False)

# 在父轴中绘制一个Inset Axes所展示区域的bbox，
# 并且绘制bbox与Inset Axes区域连线
mark_inset(ax2, axins2, loc1=2, loc2=4, fc="none", ec="0.5")

plt.show()
