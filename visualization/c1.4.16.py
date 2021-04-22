import numpy as np

import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D

from mpl_toolkits.axisartist import (
    angle_helper, Subplot, SubplotHost, ParasiteAxesAuxTrans)
from mpl_toolkits.axisartist.grid_helper_curvelinear import (
    GridHelperCurveLinear)

fig = plt.figure(figsize=(6, 5.4))

# PolarAxes.PolarTransform使用弧度，但这里需要坐标系单位为度
tr = Affine2D().scale(np.pi/180, 1) + PolarAxes.PolarTransform()
# 极坐标投影涉及到周期，在坐标上也有限制，需要一种特殊的方法来找到
# 坐标的最小值和最大值
extreme_finder = angle_helper.ExtremeFinderCycle(
        nx=20, ny=20,  # 各方向上的取样点的数量
        lon_cycle=360, lat_cycle=None,
        lon_minmax=None, lat_minmax=(0, np.inf),
        )
# 找到适合坐标的网格值（度、分、秒）
grid_locator1 = angle_helper.LocatorDMS(12)
# 使用适当的Formatter。请注意，可接受的Locator和Formatter类
# 与Matplotlib中的相应类稍有不同，后者目前还不能直接在这里使用
tick_formatter1 = angle_helper.FormatterDMS()

grid_helper = GridHelperCurveLinear(
    tr, extreme_finder=extreme_finder,
    grid_locator1=grid_locator1, tick_formatter1=tick_formatter1)
ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

# 设置右侧和上部坐标轴的刻度标签不可见
ax1.axis["right"].major_ticklabels.set_visible(True)
ax1.axis["top"].major_ticklabels.set_visible(True)
# 设置右侧坐标轴显示第一个坐标（angle）的刻度标签
ax1.axis["right"].get_helper().nth_coord_ticks = 0
# 设置左侧坐标轴显示第二个坐标（radius）的刻度标签
ax1.axis["bottom"].get_helper().nth_coord_ticks = 1

fig.add_subplot(ax1)

ax1.set_aspect(1)
ax1.set_xlim(-5, 12)
ax1.set_ylim(-5, 10)

ax1.grid(True, zorder=0)

# 由指定变换创建ParasiteAxes
ax2 = ParasiteAxesAuxTrans(ax1, tr, "equal")
# 注意：ax2.transData == tr + ax1.transData
# 任何在ax2中绘制的内容都会与ax1中的刻度相匹配
ax1.parasites.append(ax2)
ax2.plot(np.linspace(0, 30, 51), np.linspace(10, 10, 51),
        linewidth=5, color='r')

plt.show()
