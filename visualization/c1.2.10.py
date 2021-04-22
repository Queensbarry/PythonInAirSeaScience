import numpy as np
import matplotlib.pyplot as plt

z = np.random.randn(10)
rpnt, = plt.plot(z, 'ro', markersize = 13)
ypnt, = plt.plot(z[:5], '^', color = (1,1,0), markersize = 8)
plt.legend([rpnt, (rpnt, ypnt)], ['Red Points', 'Red+Yellow Points'])
plt.show()
