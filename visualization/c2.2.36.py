from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=108.)
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='coral',lake_color='aqua')
parallels = np.arange(0.,81,10.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(10.,351.,20.)
m.drawmeridians(meridians,labels=[True,False,False,True])
# 在Beijing所在位置绘制一个蓝色标记，并用文本进行标注
lon, lat = 116.46, 39.92 #Beijing的经纬度位置
# 转换至地图投影坐标，lon/lat可以是标量、列表或numpy数组
xpt,ypt = m(lon,lat)
# 坐标转换回lat/lon
lonpt, latpt = m(xpt,ypt,inverse=True)
m.plot(xpt,ypt,'b*',ms=12)  # 绘制蓝色五星
# 在五星上边偏移一些距离的位置进行文本标注，偏移量也是用地图投影坐标
plt.text(xpt,ypt-620000,'Beijing\n'+\
        '%5.1f$\degree$E,%3.1f$\degree$N'%(lonpt,latpt),\
        color='b', ha='center', va='center')
plt.show()
