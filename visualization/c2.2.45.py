import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from datetime import datetime
# mill投影
map = Basemap(projection='mill',lon_0=180)
map.drawcoastlines()
map.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
map.drawmeridians(np.arange(map.lonmin,map.lonmax+30,60),\
                    labels=[0,0,0,1])
map.drawmapboundary(fill_color='c')
map.fillcontinents(color='y',lake_color='c')
# 将夜晚区域用半透明的阴影覆盖
date = datetime.utcnow()
CS=map.nightshade(date)
plt.title('Day/Night Map for %s (UTC)' %\
           date.strftime("%d %b %Y %H:%M:%S"))
plt.show()
