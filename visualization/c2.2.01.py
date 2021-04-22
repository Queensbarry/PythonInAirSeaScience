from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
width = 28000000; lon_0 = 105; lat_0 = 40
m = Basemap(width=width,height=width,projection='aeqd',
            lat_0=lat_0,lon_0=lon_0)
# 填充背景
m.drawmapboundary(fill_color='c')
# 绘制岸线，填充陆地
m.drawcoastlines(linewidth=0.5);
m.fillcontinents(color='y',lake_color='c');
# 绘制20度间隔的网格
m.drawparallels(np.arange(-80,81,20));
m.drawmeridians(np.arange(-180,180,20));
# 在地图中心绘制红色圆点
xpt, ypt = m(lon_0, lat_0)
m.plot([xpt],[ypt],'ro')
# 绘制标题
plt.title('Azimuthal Equidistant Projection')
plt.show()
