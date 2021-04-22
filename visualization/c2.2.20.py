from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
m = Basemap(projection='gall',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=0,urcrnrlon=360,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(0.,361.,60.))
m.drawmapboundary(fill_color='c')
plt.title("Gall Stereographic Projection")
plt.show()
