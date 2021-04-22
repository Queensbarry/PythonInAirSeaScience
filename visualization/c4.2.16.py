import numpy as np
from mayavi import mlab
x, y = np.mgrid[0:3:1,0:3:1]
s = mlab.surf(x, y, np.asarray(x*0.1, 'd'))

for i in range(10):
    s.mlab_source.scalars = np.asarray(x*0.1*(i+1), 'd')
