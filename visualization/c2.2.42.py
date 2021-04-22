import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# ARGO浮标数据可以在下面的网址下载
# https://data.nodc.noaa.gov/argo/gadr/data/pacific/2017
lons,lats = np.loadtxt('argo.txt').T

# 绘制浮标位置
m = Basemap(projection='hammer',lon_0=-180)
x, y = m(lons,lats)
fig = plt.figure(figsize=(7,4))
m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#99ffff')
m.scatter(x,y,3,marker='o',color='k')
plt.title('ARGO floats in Pacific between 2017.01 and 2017.03',\
             fontsize=12)
plt.show()
