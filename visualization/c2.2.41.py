from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.colors import LightSource

# 读取 etopo5 地形数据
etopodata = Dataset('examples/etopo5.nc')
topoin = etopodata.variables['topo'][:]
lons = etopodata.variables['topo_lon'][:]
lats = etopodata.variables['topo_lat'][:]
# 将lons转换为0到360
lons[lons<0]=lons[lons<0]+360.
# 移动数据，让lons从-180变为180，而不是从0变为360
topoin,lons = shiftgrid(180.,topoin,lons,start=False)

# 将地形绘制成一幅图像
fig = plt.figure(figsize=plt.figaspect(0.45))
ax1 = fig.add_axes([0.1,0.15,0.33,0.7])
cax = fig.add_axes([0.45,0.25,0.02,0.5])
# 设置lcc投影，使用rsphere参数指定WGS84椭球
m = Basemap(llcrnrlon=34.5,llcrnrlat=1.,\
            urcrnrlon=177.434,urcrnrlat=46.352,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',area_thresh=1000.,projection='lcc',\
            lat_1=50.,lon_0=73.,ax=ax1)
# 转换本地5km间隔的nx x ny规则投影网格
nx = int((m.xmax-m.xmin)/5000.)+1; ny = int((m.ymax-m.ymin)/5000.)+1
topodat = m.transform_scalar(topoin,lons,lats,nx,ny)
# 用imshow在地图投影上绘制地形图像
im = m.imshow(topodat.data,cm.GMT_haxby)
m.drawcoastlines()
m.drawcountries()
m.drawparallels(np.arange(0.,80,20.),labels=[1,0,0,1])
m.drawmeridians(np.arange(10.,360.,30.),labels=[1,0,0,1])
cb = plt.colorbar(im,cax=cax)
ax1.set_title('ETOPO5 Topography - LCC Projection')
plt.show()

# 创建shaded relief图像
ax2 = fig.add_axes([0.57,0.15,0.33,0.7])
# 将新的Axes图像添加到已有的Basemap实例上
m.ax = ax2
# 创建光源对象
ls = LightSource(azdeg = 90, altdeg = 20)
# 将数据转换为包含光源阴影的rgb数组
try:
    rgb = ls.shade(topodat.data, cm.GMT_haxby)
except:
    rgb = ls.shade(topodat, cm.GMT_haxby)
im = m.imshow(rgb)
m.drawcoastlines()
m.drawcountries()
ax2.set_title('Shaded ETOPO5 Topography - LCC Projection')
plt.show()
