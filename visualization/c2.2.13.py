from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# 是地图左下角和右上角的 lat/lon 值
m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=0,urcrnrlon=360,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-90.,100.,30.))
m.drawmeridians(np.arange(0.,370.,60.))
m.drawmapboundary(fill_color='c')
plt.title("Equidistant Cylindrical Projection")
plt.show()
