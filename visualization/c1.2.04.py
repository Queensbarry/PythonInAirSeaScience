import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig, ax = plt.subplots()
ax.plot(100*np.random.rand(25))

formatter = ticker.FormatStrFormatter('$%1.2f')
ax.yaxis.set_major_formatter(formatter)

for tick in ax.yaxis.get_major_ticks():
    tick.tick1On = False
    tick.label1On = False
    tick.tick2On = True
    tick.label2On = True
    tick.label2.set_color('green')

plt.show()
