from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# 建立北极的极射赤面投影
# 经线lon_0处于6点钟方向, boundinglat是指在经线lon_0处与地图边缘相切的纬度
# lat_ts的默认值在极点
m = Basemap(projection='npstere',boundinglat=15,\
            lon_0=150,resolution='l')
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
plt.title("North Polar Stereographic Projection")
plt.show()
