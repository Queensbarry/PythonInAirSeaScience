import numpy as np
from mayavi import mlab

def test_quiver3d():
    x, y, z = np.mgrid[-2:3, -2:3, -2:3]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 4)
    u = y * np.sin(r) / (r + 0.001)
    v = -x * np.sin(r) / (r + 0.001)
    w = np.zeros_like(z)
    obj = mlab.quiver3d(x, y, z, u, v, w,
                        line_width=3, scale_factor=1)
    return obj

test_quiver3d()
