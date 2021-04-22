from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# 设置polyconic投影要指定区域四角的lat/lon和中心点
m = Basemap(llcrnrlon=-25.,llcrnrlat=-30,urcrnrlon=90.,urcrnrlat=50.,\
            resolution='l',area_thresh=1000.,projection='poly',\
            lat_0=0.,lon_0=30.)
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.drawmapboundary(fill_color='c')
plt.title("Polyconic Projection")
plt.show()
