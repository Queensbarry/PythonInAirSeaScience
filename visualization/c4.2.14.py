import numpy as np
from mayavi import mlab
# 创建数据
x, y = np.mgrid[-4:4:200j, -4:4:200j]
z = 100 * np.sin(x * y) / (x * y)

# 用mlab.surf进行绘图
mlab.figure(bgcolor=(1, 1, 1))
surf = mlab.surf(z, colormap='coolwarm')

# 检索surf对象的LUT
lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()

# lut是一个255x4的数组，四列数分别代表RGBA代码（red, green, blue, alpha），
# 数值为0到255之间。修改alpha通道，制造渐变的透明度
lut[:, -1] = np.linspace(0, 255, 256)

# 然后再把修改后的LUT返回给surf对象。
# 其实可以使用任意符合规定的255x4的数组，而不是要由存在的LUT进行修改
surf.module_manager.scalar_lut_manager.lut.table = lut

# 改变LUT后，强制更新绘图
mlab.draw()
mlab.view(40, 85, 420, [0, 0, 35])

mlab.show()
