import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

im1 = np.arange(100).reshape((10, 10))
im2 = im1.T
im3 = np.flipud(im1)
im4 = np.fliplr(im2)

fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # 与subplot(111)类似
                 nrows_ncols=(2, 2),  # 创建2x2的Axes网格
                 axes_pad=0.1,  # 各Axes的间距，单位为英寸
                 )

for ax, im in zip(grid, [im1, im2, im3, im4]):
    # 遍历网格，返回Axes序列的元素
    ax.imshow(im)

plt.show()
