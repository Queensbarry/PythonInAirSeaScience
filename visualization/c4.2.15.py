import numpy as np
from mayavi import mlab
# 创建间距为1的6个点
x = [1, 2, 3, 4, 5, 6]
y = [0, 0, 0, 0, 0, 0]
z = y
# 提供一个变化为0.5到1的标量
s = [.5, .6, .7, .8, .9, 1]
# 使用points3d进行绘图
mlab.figure(bgcolor=(1, 1, 1), size=(400,150))
pts = mlab.points3d(x, y, z, s)
mlab.view(-90, 90, 4, [3.5, 0, 0])
