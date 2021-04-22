import numpy as np
import matplotlib.pyplot as plt

plt.subplot(211)
plt.plot([1, 2, 3], 'r-', label='line 1')
plt.plot([3, 2, 1], 'b:', label='line 2')
# 将图例放在子图上方，而且图例宽度与子图宽度一样
plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.1), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)

plt.subplot(223)
plt.plot([1, 2, 3], 'r-', label='line 1')
plt.plot([3, 2, 1], 'b:', label='line 2')
# 将图例放在该子图右边
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
