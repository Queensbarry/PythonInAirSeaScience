import numpy as np
from mayavi import mlab

def test_barchart():
    """ Demo the bar chart plot with a 2D array."""
    s = np.abs(np.random.random((3, 3)))
    return mlab.barchart(s)

test_barchart()
