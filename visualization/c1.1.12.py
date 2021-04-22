import matplotlib.pyplot as plt
import numpy as np

# 构造数据
N = 5
Means = (21, 35, 30, 32, 27)
Std = (3, 2, 4, 1, 2)

ind = np.arange(N)    # 设置各组在x轴的位置
width = 0.45       # 每个条形的宽度，也可以用一个长度为N的序列来指定

p = plt.bar(ind, Means, width, yerr=Std, capsize=3)

plt.ylabel('Scores')
plt.title('Scores by group')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 41, 10))

plt.show()
