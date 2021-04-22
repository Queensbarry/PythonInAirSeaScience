import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.basemap import Basemap
from scipy.ndimage.filters import minimum_filter, maximum_filter
from netCDF4 import Dataset

def extrema(mat,mode='wrap',window=10):
    """find the indices of local extrema (min and max)
    in the input array."""
    mn = minimum_filter(mat, size=window, mode=mode)
    mx = maximum_filter(mat, size=window, mode=mode)
    return np.nonzero(mat == mn), np.nonzero(mat == mx)

# 当日时间
date = '20210319'

# OpenDAP数据集
url="https://nomads.ncep.noaa.gov/dods/gfs_1p00/gfs"+\
     date+"/gfs_1p00_00z"
data=Dataset(url)

# 读取lats,lons
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]
# 读取prmsl，转换为hPa (mb).
prmsl = 0.01*data.variables['prmslmsl'][0]
# 参数window控制可探测到的高低压数量，取值越大，能找到高低压数量越少
local_min, local_max = extrema(prmsl, mode='wrap', window=50)
m = Basemap(llcrnrlon=0,llcrnrlat=-80,\
            urcrnrlon=360,urcrnrlat=80,projection='mill')
# 设置要绘制的等值线列表
clevs = np.arange(900,1100.,5.)
# 创建投影网格
lons, lats = np.meshgrid(lons, lats)
x, y = m(lons, lats)

fig=plt.figure(figsize=(8,4.5))
ax = fig.add_axes([0.05,0.05,0.9,0.85])
cs = m.contour(x,y,prmsl,clevs,colors='k',linewidths=1.)
m.drawcoastlines(linewidth=1.25)
m.fillcontinents(color='0.8')
m.drawparallels(np.arange(-80,81,20),labels=[1,1,0,0])
m.drawmeridians(np.arange(0,360,60),labels=[0,0,0,1])
xlows = x[local_min]; xhighs = x[local_max]
ylows = y[local_min]; yhighs = y[local_max]
lowvals = prmsl[local_min]; highvals = prmsl[local_max]
# 用蓝色“L”标注低压，并在其下方标注数值
xyplotted = []
# 如果已经有了L或H的标注，则不再进行绘制
yoffset = 0.022*(m.ymax-m.ymin)
dmin = yoffset
for x,y,p in zip(xlows, ylows, lowvals):
    if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
        if not dist or min(dist) > dmin:
            plt.text(x,y,'L',fontsize=14,fontweight='bold',
                    ha='center',va='center',color='b')
            plt.text(x,y-yoffset,repr(int(p)),fontsize=9,
                    ha='center',va='top',color='b',
                    bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
            xyplotted.append((x,y))
# 用红色“H”标注高压，并在其下方标注数值
xyplotted = []
for x,y,p in zip(xhighs, yhighs, highvals):
    if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
        if not dist or min(dist) > dmin:
            plt.text(x,y,'H',fontsize=14,fontweight='bold',
                    ha='center',va='center',color='r')
            plt.text(x,y-yoffset,repr(int(p)),fontsize=9,
                    ha='center',va='top',color='r',
                    bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
            xyplotted.append((x,y))
plt.title('Mean Sea-Level Pressure (with Highs and Lows) %s' % date)
plt.show()
