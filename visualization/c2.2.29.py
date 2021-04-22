from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# 建立北极兰伯特方位投影
m = Basemap(projection='nplaea',boundinglat=15,\
            lon_0=170,resolution='l')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(0.,361.,20.))
m.drawmapboundary(fill_color='c')
ax = plt.gca()
for y in np.linspace(m.ymax/20,19*m.ymax/20,10):
    for x in np.linspace(m.xmax/20,19*m.xmax/20,10):
        lon, lat = m(x,y,inverse=True)
        poly = m.tissot(lon,lat,2.5,100,\
                        facecolor='red',zorder=10,alpha=0.5)
plt.title("North Polar Lambert Azimuthal Projection")
plt.show()
