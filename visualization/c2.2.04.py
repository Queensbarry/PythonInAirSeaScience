from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# lon_0为投影的中心经度
# 可选参数 'satellite_height' 用于指定同步轨道的高度（默认为35,786公里）
# rsphere=(6378137.00,6356752.3142)指定了WGS84投影椭球
m = Basemap(projection='geos',\
            rsphere=(6378137.00,6356752.3142),\
            lon_0=105,resolution='l')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-90.,100.,30.))
m.drawmeridians(np.arange(0.,370.,30.))
m.drawmapboundary(fill_color='c')
plt.title("Full Disk Geostationary Projection")
plt.show()
