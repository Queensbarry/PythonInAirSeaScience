import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axisartist import GridHelperCurveLinear

def curvelinear_test(fig):
    """Polar projection, but in a rectangular box.
    """
    # 创建一个极坐标变换。PolarAxes.PolarTransform使用弧度，但本例
    # 要设置的坐标系中角度的单位为度
    tr = Affine2D().scale(np.pi / 180., 1.) + PolarAxes.PolarTransform()

    # 极坐标投影涉及到周期，在坐标上也有限制，需要一种特殊的方法来找到
    # 坐标的最小值和最大值
    extreme_finder = angle_helper.ExtremeFinderCycle(20,
                                                     20,
                                                     lon_cycle=360,
                                                     lat_cycle=None,
                                                     lon_minmax=None,
                                                     lat_minmax=(0,
                                                                 np.inf),
                                                     )
    # 找到适合坐标的网格值（度、分、秒）
    grid_locator1 = angle_helper.LocatorDMS(12)

    # 使用适当的Formatter。请注意，可接受的Locator和Formatter类
    # 与Matplotlib中的相应类稍有不同，后者目前还不能直接在这里使用
    tick_formatter1 = angle_helper.FormatterDMS()

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )

    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

    fig.add_subplot(ax1)

    # 创建浮动坐标轴

    # 浮动坐标轴的第一个坐标（theta）指定为60度
    ax1.axis["lat"] = axis = ax1.new_floating_axis(0, 60)
    axis.label.set_text(r"$\theta = 60^{\circ}$")
    axis.label.set_visible(True)

    # 浮动坐标轴的第二个坐标（r）指定为6
    ax1.axis["lon"] = axis = ax1.new_floating_axis(1, 6)
    axis.label.set_text(r"$r = 6$")

    ax1.set_aspect(1.)
    ax1.set_xlim(-5, 12)
    ax1.set_ylim(-5, 10)

    ax1.grid(True)

fig = plt.figure(figsize=(5, 5))
curvelinear_test(fig)
plt.show()
