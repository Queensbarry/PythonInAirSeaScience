import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

# 给y指定一列在[0, 1]区间的数
y = np.arange(0.01,1.,0.01)
x = np.arange(len(y))

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True, linestyle=':')

# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True, linestyle=':')

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.01)
plt.title('symlog')
plt.grid(True, linestyle=':')

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True, linestyle=':')

# 调整各子图的显示以及logit图中y轴刻度标签的格式
plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,
                    hspace=0.25, wspace=0.35)

plt.show()
