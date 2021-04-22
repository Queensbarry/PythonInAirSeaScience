from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
# 使用SODA数据
dataset = Dataset('data/soda224sfc2010.nc')
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:] - 360.
lon, lat = np.meshgrid(lon,lat)
sst = dataset.variables['temp'][0,:].squeeze()

fig = plt.figure()
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(projection='kav7',lon_0=-180,resolution=None)
m.drawmapboundary(fill_color='0.3')
# 绘制SST
im = m.pcolormesh(lon,lat,sst,shading='flat',\
                    cmap=plt.cm.jet,latlon=True)
m.drawparallels(np.arange(-90.,99.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
# colorbar
cb = m.colorbar(im,"bottom", size="5%", pad="2%")
# title.
ax.set_title('SST analysis for 2010.01')
plt.show()
