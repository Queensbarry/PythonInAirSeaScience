from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# satellite_height 是相机（视角）所处的高度
h = 3000.
m = Basemap(projection='nsper',lon_0=105,lat_0=30,
        satellite_height=h*1000.,resolution='l')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-90.,100.,30.))
m.drawmeridians(np.arange(0.,370.,30.))
m.drawmapboundary(fill_color='c')
plt.title("Full Disk Near-Sided Perspective Projection " +
        "%d km above earth" % h, fontsize=10)
plt.show()
