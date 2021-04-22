import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

def get_demo_image():
    import numpy as np
    x,y = np.meshgrid(np.linspace(-1,1,15),
                      np.linspace(-1,1,15))
    z = (x+y)*np.exp(-5*(x**2+y**2))*6.01-0.152
    # z is a numpy array of 15x15
    return z, (-3, 4, -4, 3)

def get_rgb():
    Z, extent = get_demo_image()

    Z[Z < 0] = 0.
    Z = Z / Z.max()

    R = Z[:13, :13]
    G = Z[2:, 2:]
    B = Z[:13, 2:]

    return R, G, B

fig = plt.figure()
ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])

r, g, b = get_rgb()
kwargs = dict(origin="lower", interpolation="nearest")
ax.imshow_rgb(r, g, b, **kwargs)

ax.RGB.set_xlim(2., 11.5)
ax.RGB.set_ylim(2.9, 12.4)

plt.show()
