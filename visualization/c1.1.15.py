import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3)
y1 = (1.0 + 0.5 * x) ** 3
y2 = (2.5 - 0.5 * x) ** 3
y3 = y1 + y2
line1, line2, line3 = plt.plot(x, y1, 'k--',
                               x, y2, 'k:',
                               x, y3, 'k-', lw=2)
plt.legend((line1, line2, line3), ('Line 1', 'Line 2', 'Line 3'),
           loc='upper center', fontsize='x-large',facecolor='C7')
plt.show()
