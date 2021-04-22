from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# width（xmax-xmin，km）是投影区域的宽度
# height（ymax-ymin，km）是投影区域的高度
# lat_1、lon_1和lat_2、lon_2是一对用来定义投影中心线的点
# 地图投影会自动旋转至真北方向，可以通过设置no_rot=True为避免此行为
# area_thresh=1000 表示岸线特征面积小于1000km^2的海岸线将不会被绘制
m = Basemap(height=16700000,width=12000000,\
            resolution='l',area_thresh=1000.,projection='omerc',\
            lon_0=100,lat_0=15,lon_2=80,lat_2=65,lon_1=150,lat_1=-55)
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(0.,361.,20.))
m.drawmapboundary(fill_color='c')
plt.title("Oblique Mercator Projection")
plt.show()
