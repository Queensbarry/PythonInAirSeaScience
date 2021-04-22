import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def set_demo_image():
    import numpy as np
    x,y = np.meshgrid(np.linspace(-1,1,15),
                      np.linspace(-1,1,15))
    z = (x+y)*np.exp(-5*(x**2+y**2))*6.01-0.152
    return z, (-3, 4, -4, 3)

def demo_simple_grid(fig):
    """
    A grid of 2x2 images with 0.05 inch pad between images and only
    the lower-left axes is labeled.
    """
    grid = ImageGrid(fig, 221,  # 与subplot(221)类似
                     nrows_ncols=(2, 2),
                     axes_pad=0.05,
                     label_mode="1",
                     )

    Z, extent = set_demo_image()
    for ax in grid:
        im = ax.imshow(Z, extent=extent, interpolation="nearest")

    # 因share_all=False，下面操作只对第一列和第二行的Axes起作用
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])

def demo_grid_with_single_cbar(fig):
    """
    A grid of 2x2 images with a single colorbar
    """
    grid = ImageGrid(fig, 222,  # 与subplot(222)类似
                     nrows_ncols=(2, 2),
                     axes_pad=0.0,
                     share_all=True,
                     label_mode="L",
                     cbar_location="top",
                     cbar_mode="single",
                     )

    Z, extent = set_demo_image()
    for ax in grid:
        im = ax.imshow(Z, extent=extent, interpolation="nearest")
    grid.cbar_axes[0].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(False)

    # 因share_all=True，下面的操作对所有Axes有效
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])

def demo_grid_with_each_cbar(fig):
    """
    A grid of 2x2 images. Each image has its own colorbar.
    """
    grid = ImageGrid(fig, 223,  # 类似subplot(223)
                     nrows_ncols=(2, 2),
                     axes_pad=0.1,
                     label_mode="1",
                     share_all=True,
                     cbar_location="top",
                     cbar_mode="each",
                     cbar_size="7%",
                     cbar_pad="2%",
                     )
    Z, extent = set_demo_image()
    for ax, cax in zip(grid, grid.cbar_axes):
        im = ax.imshow(Z, extent=extent, interpolation="nearest")
        cax.colorbar(im)
        cax.toggle_label(False)

    # 因share_all=True，下面操作对所有Axes有效
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])

def demo_grid_with_each_cbar_labelled(fig):
    """
    A grid of 2x2 images. Each image has its own colorbar.
    """
    grid = ImageGrid(fig, 224,  # 类似subplot(224)
                     nrows_ncols=(2, 2),
                     axes_pad=(0.45, 0.15),
                     label_mode="1",
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="each",
                     cbar_size="7%",
                     cbar_pad="2%",
                     )
    Z, extent = set_demo_image()

    # 对网格中各colorbar使用不同的范围
    limits = ((0, 1), (-2, 2), (-1.7, 1.4), (-1.5, 1))
    for ax, cax, vlim in zip(grid, grid.cbar_axes, limits):
        im = ax.imshow(Z, extent=extent, interpolation="nearest",
                       vmin=vlim[0], vmax=vlim[1])
        cax.colorbar(im)
        cax.set_yticks((vlim[0], vlim[1]))

    # 因share_all=True，下面操作对所有Axes有效
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0.07, right=0.93,
                    bottom=0.07, top=0.93)

demo_simple_grid(fig)
demo_grid_with_single_cbar(fig)
demo_grid_with_each_cbar(fig)
demo_grid_with_each_cbar_labelled(fig)

plt.show()
