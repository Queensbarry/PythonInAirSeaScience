from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=108.)
# 绘制海岸线
m.drawcoastlines()
# 绘制地图边框，并填充颜色
m.drawmapboundary(fill_color='aqua')
# 填充陆地，并用海洋颜色填充湖泊
m.fillcontinents(color='coral',lake_color='aqua')
# 绘制经纬线，并在右侧和顶侧标注纬度，在底部和左侧标注经度
parallels = np.arange(0.,81,10.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(10.,351.,20.)
m.drawmeridians(meridians,labels=[True,False,False,True])
plt.show()
