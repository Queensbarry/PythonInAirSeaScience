from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# 设置 Lambert Conformal 投影
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=108.)
# 绘制海岸线
m.drawcoastlines()
# 给地图添加边框，并填充背景，此背景为海洋的颜色，而陆地会被画在其之上
m.drawmapboundary(fill_color='aqua')
# 填充陆地，并设置湖泊采用与海洋相同的颜色
m.fillcontinents(color='coral',lake_color='aqua')
plt.show()

# 设置 Lambert Conformal 投影
# 设置resolution=None来略过边界数据集的处理
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=108.)
# 绘制陆-海掩模作为地图背景
# lakes=True表示用海洋颜色绘制内陆湖泊
fig=plt.figure()
m.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)
plt.show()

fig=plt.figure()
m.bluemarble()
plt.show()

fig=plt.figure()
m.shadedrelief()
plt.show()

fig=plt.figure()
m.etopo()
plt.show()
