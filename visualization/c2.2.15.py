from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# lat_ts是真实标度的纬度
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=0,urcrnrlon=360,lat_ts=20,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(0.,361.,60.))
m.drawmapboundary(fill_color='c')
plt.title("Mercator Projection")
plt.show()
