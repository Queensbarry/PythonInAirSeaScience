from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
m = Basemap(width=15.e6,height=15.e6,\
            projection='gnom',lat_0=60.,lon_0=120.)
m.drawmapboundary(fill_color='c')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(10,90,20))
m.drawmeridians(np.arange(-180,180,30))
plt.title('Gnomonic Projection')
plt.show()
