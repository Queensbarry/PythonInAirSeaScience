import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
m = Basemap(llcrnrlon=-100.,llcrnrlat=0.,\
            urcrnrlon=-20.,urcrnrlat=57.,\
            resolution ='l',area_thresh=1000.)
# 读取shapefile
shp_info = m.readshapefile('data/huralll020',\
                        'hurrtracks',drawbounds=False)
# 查找强度达到4级的飓风的名称
names = []
for shapedict in m.hurrtracks_info:
    cat = shapedict['CATEGORY']
    name = shapedict['NAME']
    if cat in ['H4','H5'] and name not in names:
        # 只取有名字的飓风
        if name != 'NOT NAMED':  names.append(name)
# 绘制飓风轨迹
for shapedict,shape in zip(m.hurrtracks_info,m.hurrtracks):
    name = shapedict['NAME']
    cat = shapedict['CATEGORY']
    if name in names:
        xx,yy = zip(*shape)
        # 用红色绘制飓风轨迹中超过4级的部分
        if cat in ['H4','H5']:
            m.plot(xx,yy,linewidth=1.5,color='r')
        elif cat in ['H1','H2','H3']:
            m.plot(xx,yy,color='k')
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#99ffff')
m.drawparallels(np.arange(10,70,20),labels=[1,1,0,0])
m.drawmeridians(np.arange(-100,0,20),labels=[0,0,0,1])
plt.title('Atlantic Hurricane Tracks (Storms Reaching Category 4, 1851-2004)')
plt.show()
