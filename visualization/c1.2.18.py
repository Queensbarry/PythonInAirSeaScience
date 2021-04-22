import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure()
opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
            ha='center', va='center', fontsize=14)
ax1 = plt.subplot2grid((3, 3), (0, 0)) \
                        .annotate('ax1', **opts)
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2) \
                        .annotate('ax2', **opts)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2) \
                        .annotate('ax3', **opts)
ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) \
                        .annotate('ax4', **opts)
plt.tight_layout()

plt.show()
