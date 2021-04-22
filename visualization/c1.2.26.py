import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms

fig, ax = plt.subplots()
x = np.random.randn(1000)

ax.hist(x, 30)
ax.set_title(r'$\sigma=1 \/ \dots \/ \sigma=2$', fontsize=16)

# 将x坐标转换为数据坐标，保持y坐标为Axes坐标
trans = transforms.blended_transform_factory(
    ax.transData, ax.transAxes)

# 用一个跨度来突出显示1..2区间
# 这里要让跨度在x方向处于数据坐标，而y方向则以Axes坐标覆盖从0到1的范围
rect = mpatches.Rectangle((1, 0), width=1, height=1,
                         transform=trans, color='yellow',
                         alpha=0.5)

ax.add_patch(rect)

plt.show()
