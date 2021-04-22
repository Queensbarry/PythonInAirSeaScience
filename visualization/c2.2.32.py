from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# 建立南极方位等距投影
m = Basemap(projection='spaeqd',boundinglat=-15,\
            lon_0=130,resolution='l')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(0.,361.,20.))
m.drawmapboundary(fill_color='c')
ax = plt.gca()
for y in np.linspace(19*m.ymin/20,m.ymin/20,10):
    for x in np.linspace(19*m.xmin/20,m.xmin/20,10):
        lon, lat = m(x,y,inverse=True)
        poly = m.tissot(lon,lat,2.5,100,\
                        facecolor='red',zorder=10,alpha=0.5)
plt.title("South Polar Azimuthal Equidistant Projection")
plt.show()
