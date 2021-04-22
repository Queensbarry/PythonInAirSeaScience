import numpy as np
from mayavi import mlab
# 创建数据
n_mer, n_long = 6, 11
pi = np.pi
dphi = pi/1000.0
phi = np.arange(0.0, 2*pi + 0.5*dphi, dphi, 'd')
mu = phi*n_mer
x = np.cos(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
y = np.sin(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
z = np.sin(n_long*mu/n_mer)*0.5

l = mlab.plot3d(x, y, z, np.sin(mu),
                tube_radius=0.025, colormap='Spectral')

ms = l.mlab_source
@mlab.animate()
def anim():
    for i in range(10):
        x = np.cos(mu)*(1+np.cos(n_long*mu/n_mer +
                                      np.pi*(i+1)/5.)*0.5)
        scalars = np.sin(mu + np.pi*(i+1)/5)
        ms.trait_set(x=x, scalars=scalars)
        yield

anim()
