import numpy as np
import matplotlib.pyplot as plt

ln1, = plt.plot([1, 2, 3], 'r-', label='line 1')
ln2, = plt.plot([3, 2, 1], 'b:', label='line 2')

# 为ln1创建图例
lg1 = plt.legend(handles=[ln1], loc=1)
# 将lg1添加到当前Axes
ax1 = plt.gca().add_artist(lg1)

# 为ln2创建图例
lg2 = plt.legend(handles=[ln2], loc=4)
# 将lg2添加到当前Axes
ax2 = plt.gca().add_artist(lg2)

plt.show()
