from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# lat_1、lat_2 分别是第一、二标准纬线，lat_2默认值等于lat_1
# rsphere=(6378137.00,6356752.3142)指定了WGS84投影椭球
m = Basemap(width=12000000,height=9000000,
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',area_thresh=1000.,projection='lcc',\
            lat_1=45.,lat_2=55,lat_0=50,lon_0=108.)
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(0.,361.,20.))
m.drawmapboundary(fill_color='c')
# 绘制tissot's indicatrix以展示失真程度
ax = plt.gca()
for y in np.linspace(m.ymax/20,19*m.ymax/20,9):
    for x in np.linspace(m.xmax/20,19*m.xmax/20,12):
        lon, lat = m(x,y,inverse=True)
        poly = m.tissot(lon,lat,2.,100,\
                        facecolor='red',zorder=10,alpha=0.5)
plt.title("Lambert Conformal Projection")
plt.show()
