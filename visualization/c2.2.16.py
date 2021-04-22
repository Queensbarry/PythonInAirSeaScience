from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
m = Basemap(llcrnrlon=152.63,llcrnrlat=49.55,
            urcrnrlon=168.33,urcrnrlat=59.55,
            resolution='i',projection='tmerc',lon_0=159.5,lat_0=54.8)
# 也可以用下面指定width和height的方法来得到同样的地图
#m = Basemap(width=992500,height=1126400,\
#            resolution='i',projection='tmerc',lon_0=159.5,lat_0=54.8)
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(48,61.,2.))
m.drawmeridians(np.arange(150.,170,2.))
m.drawmapboundary(fill_color='c')
plt.title("Transverse Mercator Projection")
plt.show()

