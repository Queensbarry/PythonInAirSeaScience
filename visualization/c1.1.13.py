import matplotlib.pyplot as plt
import numpy as np

labels = 'Red', 'Green', 'Blue', 'Gray'
sizes = [17, 26, 44, 13]
colors = ['red', 'green', 'blue', 'gray']
explode = (0, 0.1, 0, 0)  # 把第二分区的扇形进行强调显示

fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors=colors, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # 等长径比确保饼图被画成圆形

plt.show()
