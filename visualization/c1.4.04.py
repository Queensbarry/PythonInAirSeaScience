import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 随机数据
x = np.random.randn(1000)
y = np.random.randn(1000)

fig, axScatter = plt.subplots(figsize=(5.5, 5.5))

# 绘制散点图
axScatter.scatter(x, y)
axScatter.set_aspect(1.)

# 在当前Axes的右侧和顶部创建新的Axes，append_axes的第二个参数分别为
# 顶部水平放置的新Axes的高度和右侧垂直放置的新Axes的宽度，单位为英寸
divider = make_axes_locatable(axScatter)
axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

# 设置某些标签文本不可见
axHistx.xaxis.set_tick_params(labelbottom=False)
axHisty.yaxis.set_tick_params(labelleft=False)

# 手动设置适合的范围
binwidth = 0.25
xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
lim = (int(xymax/binwidth) + 1)*binwidth

bins = np.arange(-lim, lim + binwidth, binwidth)
axHistx.hist(x, bins=bins)
axHisty.hist(y, bins=bins, orientation='horizontal')

# axHistx的x轴和axHisty的y轴都与axScatter共享，
# 这样就不需要手动调整它们的xlim和ylim
axHistx.set_yticks([0, 50, 100])
axHisty.set_xticks([0, 50, 100])

plt.show()
