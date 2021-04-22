import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import datetime
import matplotlib.dates as mdates

x = np.linspace(0.0, 5.0, 100)
y = np.cos(2 * np.pi * x) * np.exp(-x)

fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
base = datetime.datetime(2019, 6, 1, 0, 0, 0)
time = [base + datetime.timedelta(days=t) for t in range(len(y))]

ax.plot(time, y)
ax.tick_params(axis='x', rotation=70)
plt.show()


locator = mdates.DayLocator(bymonthday=[1, 15])
formatter = mdates.DateFormatter('%b %d')
fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.plot(time, y)
ax.tick_params(axis='x', rotation=70)
plt.show()
