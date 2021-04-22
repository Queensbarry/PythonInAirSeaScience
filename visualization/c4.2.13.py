import numpy as np
from mayavi import mlab

def test_volume_slice():
    x, y, z = np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]
    scalars = x * x * 0.5 + y * y + z * z * 2.0
    obj = mlab.volume_slice(scalars,
                            plane_orientation='x_axes')
    return obj

test_volume_slice()
