from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
m = Basemap(width=8000000,height=7000000,\
            resolution='l',projection='aea',\
            lat_1=40.,lat_2=60,lon_0=65,lat_0=50)
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(0.,361.,20.))
m.drawmapboundary(fill_color='c')
ax = plt.gca()
for y in np.linspace(m.ymax/20,19*m.ymax/20,10):
    for x in np.linspace(m.xmax/20,19*m.xmax/20,12):
        lon, lat = m(x,y,inverse=True)
        poly = m.tissot(lon,lat,1.6,100,\
                        facecolor='red',zorder=10,alpha=0.5)
plt.title("Albers Equal Area Projection")
plt.show()
