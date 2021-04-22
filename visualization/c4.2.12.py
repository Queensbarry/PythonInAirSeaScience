import numpy as np
from mayavi import mlab

def test_flow():
    x, y, z = np.mgrid[-4:4:40j, -4:4:40j, 0:4:20j]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2 + 0.1)
    u = y * np.sin(r) / r
    v = -x * np.sin(r) / r
    w = np.ones_like(z)*0.05
    obj = mlab.flow(u, v, w)
    return obj

test_flow()
