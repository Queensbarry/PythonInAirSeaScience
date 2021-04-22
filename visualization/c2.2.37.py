from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
map = Basemap(projection='ortho',lat_0=45,lon_0=105,resolution='l')
# 绘制海岸线并填充陆地
map.drawcoastlines(linewidth=0.25)
# map.drawcountries(linewidth=0.25) # 国界线
map.fillcontinents(color='y',lake_color='c')
map.drawmapboundary(fill_color='c')
map.drawmeridians(np.arange(0,360,30))
map.drawparallels(np.arange(-90,90,30))
# 创建数据
nlats = 73; nlons = 145; delta = 2.*np.pi/(nlons-1)
lats = (0.5*np.pi-delta*np.indices((nlats,nlons))[0,:,:])
lons = (delta*np.indices((nlats,nlons))[1,:,:])
wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)
# 坐标转换
x, y = map(lons*180./np.pi, lats*180./np.pi)
# 在地图上绘制等值线
cs = map.contour(x,y,wave+mean,15,lw=2,cmap=plt.cm.jet)
plt.title('contour lines over filled continent background')
plt.show()
