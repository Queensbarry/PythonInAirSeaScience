import numpy as np
from mayavi import mlab

def test_points3d():
    t = np.linspace(0, 4 * np.pi, 20)

    x = np.sin(2 * t)
    y = np.cos(t)
    z = np.cos(2 * t)
    s = 2 + np.sin(t)

    return mlab.points3d(x, y, z, s,
                        colormap="copper", scale_factor=.25)

test_points3d()
