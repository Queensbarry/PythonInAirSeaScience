import matplotlib.pyplot as plt
import numpy as np

mu, sigma = 100, 15
x = mu + sigma * np.random.standard_normal(9000)

# 根据上述数据绘制直方图
n, bins, patches = plt.hist(x, 45, density=True, facecolor='k', alpha=0.6)

# 添加文本
plt.xlabel('Intelligence Quotient')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(130, .027, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()
