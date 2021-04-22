from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# 是地图左下角和右上角的 lat/lon 值
# resolution = 'i' 表示用中等分辨率（intermediate resolution）的岸线数据
# lon_0, lat_0 为投影的中心经、纬度
m = Basemap(llcrnrlon=152.63,llcrnrlat=49.55,
            urcrnrlon=168.33,urcrnrlat=59.55,
            resolution='i',projection='cass',lon_0=159.5,lat_0=54.8)
# 也可以用下面指定width和height的方法来得到同样的地图
#m = Basemap(width=988700,height=1125200,\
#            resolution='i',projection='cass',lon_0=159.5,lat_0=54.8)
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
# 绘制经纬线
m.drawparallels(np.arange(48,61.,2.))
m.drawmeridians(np.arange(150.,170,2.))
m.drawmapboundary(fill_color='c')
plt.title("Cassini Projection")
plt.show()
