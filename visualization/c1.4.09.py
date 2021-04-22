import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, (ax, ax2) = plt.subplots(1, 2, figsize=[5.5, 2.8])

# 创建宽度为1.3英寸，高度为0.9英寸的Inset Axes，默认位置为右上角
axins = inset_axes(ax, width=1.3, height=0.9)

# 在左下角(loc=3)创建宽度和高度分别为父轴的30%和40%的Inset Axes
axins2 = inset_axes(ax, width="30%", height="40%", loc=3)

# 在第二个子图中用混合指定的方式创建Inset Axes
# 宽度为父轴的30%，高度为1英寸，位置为左上角(loc=2)
axins3 = inset_axes(ax2, width="30%", height=1., loc=2)

# 在右下角(loc=4)创建一个Inset Axes，指定borderpad=1，
# 即其与父轴轴线的距离为1个字体大小，即10个点（默认字体尺寸为10pt）
axins4 = inset_axes(ax2, width="20%", height="20%", loc=4, borderpad=1)

# 关闭显示Inset Axes的刻度标签
for axi in [axins, axins2, axins3, axins4]:
    axi.tick_params(labelleft=False, labelbottom=False)

plt.show()
