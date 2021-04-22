import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.grid_helper_curvelinear \
     import GridHelperCurveLinear
from mpl_toolkits.axisartist import Subplot

# 从曲线坐标到直角坐标
def tr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return x, y-x

# 从直角坐标到曲线坐标
def inv_tr(x,y):
    x, y = np.asarray(x), np.asarray(y)
    return x, y+x

grid_helper = GridHelperCurveLinear((tr, inv_tr))

fig = plt.figure()
ax1 = Subplot(fig, 1, 1, 1, grid_helper=grid_helper)

fig.add_subplot(ax1)
ax1.grid('on')

plt.show()
