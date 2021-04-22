from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,3))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
m = Basemap(llcrnrlon=-10.,llcrnrlat=25.,\
            urcrnrlon=125.,urcrnrlat=70.)
# 北京经纬度
bjlat = 39.92; bjlon = 116.46
# 伦敦经纬度
lonlat = 51.53; lonlon = 0.08
# 绘制两者之间的大圆
m.drawgreatcircle(lonlon,lonlat,bjlon,bjlat,linewidth=2,color='b')
m.drawcoastlines()
m.fillcontinents()
m.drawparallels(np.arange(10,90,20),labels=[1,1,0,1])
m.drawmeridians(np.arange(0,360,30),labels=[1,1,0,1])
ax.set_title('Great Circle from Beijing to London')
plt.show()
