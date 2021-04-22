import numpy as np
from mayavi import mlab

def test_contour3d():
    x, y, z = np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]

    scalars = x * x * 0.5 + y * y + z * z * 2.0

    obj = mlab.contour3d(scalars, contours=4, transparent=True)
    return obj

test_contour3d()
