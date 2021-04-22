from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# lon_0, lat_0 为投影的中心点
# resolution = 'l' 表示用低分辨率（low resolution）的岸线数据
m = Basemap(projection='ortho',lon_0=105,lat_0=30,resolution='l')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
# 绘制经纬线
m.drawparallels(np.arange(-90.,100.,30.))
m.drawmeridians(np.arange(0.,370.,30.))
m.drawmapboundary(fill_color='c')
plt.title("Full Disk Orthographic Projection")
plt.show()
