import matplotlib.pyplot as plt

fig, axs = plt.subplots(1,2,figsize=(6,2.5))
axs[0].plot(0.5, 0.5, 'ro', ms=8)
axs[1].annotate("Test", size=16, color='blue',
                  xy=(0.5, 0.5), xycoords=axs[0].transData,
                  xytext=(0.5, 0.5), textcoords=axs[1].transData,
                  arrowprops=dict(arrowstyle="->", shrinkB=5))
plt.show()

fig, ax = plt.subplots(figsize=(4,3))
an1 = ax.annotate("Test 1", xy=(0.5, 0.5), xycoords="data",
                  va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))
an2 = ax.annotate("Test 2", xy=(1, 0.5), xycoords=an1,
                  xytext=(30,0), textcoords="offset points",
                  va="center", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))
plt.show()

fig, ax = plt.subplots(figsize=(4, 2.5))
an1 = ax.annotate("Test 1", xy=(0.5, 0.5), xycoords="data",
                  va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))

an2 = ax.annotate("Test 2", xy=(0.5, 1.), xycoords=an1,
                  xytext=(0.5, 1.1), textcoords=(an1, "axes fraction"),
                  va="bottom", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))

fig.subplots_adjust(top=0.83)
plt.show()

from matplotlib.text import OffsetFrom

fig, ax = plt.subplots(figsize=(4, 2.5))
ax.plot(0.1, 0.1, 'ro')
ax.axis([0, 1, 0, 1])
an1 = ax.annotate("Test 1", xy=(0.6, 0.6), xycoords="data",
                  va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))

offset_from = OffsetFrom(an1, (0.5, 0))
an2 = ax.annotate("Test 2", xy=(0.1, 0.1), xycoords="data",
                  xytext=(0, -10), textcoords=offset_from,
                  va="top", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))
plt.show()
