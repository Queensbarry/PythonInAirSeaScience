from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# lon_0为投影的中心经度
# resolution = 'c' 表示用粗分辨率（crude resolution）的岸线数据
m = Basemap(projection='moll',lon_0=-180,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
# 绘制经纬线
m.drawparallels(np.arange(-90.,100.,30.))
m.drawmeridians(np.arange(0.,370.,60.))
m.drawmapboundary(fill_color='c')
plt.title("Mollweide Projection")
plt.show()

