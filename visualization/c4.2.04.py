import numpy as np
from mayavi import mlab

def test_imshow():
    """ Use imshow to visualize a 2D 10x10 random array."""
    s = np.random.random((10, 10))
    return mlab.imshow(s, colormap='gist_earth')

test_imshow()
