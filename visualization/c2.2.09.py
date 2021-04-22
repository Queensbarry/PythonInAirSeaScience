from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
m = Basemap(projection='eck4',lon_0=-180,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='y',lake_color='c')
m.drawparallels(np.arange(-90.,100.,30.))
m.drawmeridians(np.arange(0.,370.,60.))
m.drawmapboundary(fill_color='c')
plt.title("Eckert IV Projection")
plt.show()
