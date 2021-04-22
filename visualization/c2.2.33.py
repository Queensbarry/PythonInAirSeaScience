from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
m = Basemap(projection='vandg',lon_0=180,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(0.,361.,60.))
m.drawmapboundary(fill_color='c')
plt.title("van der Grinten Projection")
plt.show()
