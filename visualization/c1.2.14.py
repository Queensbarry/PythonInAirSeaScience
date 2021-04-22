import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig, axs = plt.subplots(ncols = 2, nrows = 2, constrained_layout = True)

fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[1, 0])
ax4 = fig.add_subplot(spec[1, 1])

plt.show()

fig = plt.figure(constrained_layout=True) 
gs = fig.add_gridspec(3, 3)
opts = dict(ha = 'center', va = 'center')
ax1 = fig.add_subplot(gs[0, :])
ax1.text(0.5, 0.5, 'ax1\ngs[0, :]', **opts)
ax2 = fig.add_subplot(gs[1, : - 1])
ax2.text(0.5, 0.5, 'ax2\ngs[1, :-1]', **opts)
ax3 = fig.add_subplot(gs[1:, - 1])
ax3.text(0.5, 0.5, 'ax3\ngs[1:, -1]', **opts)
ax4 = fig.add_subplot(gs[ - 1, 0]) 
ax4.text(0.5, 0.5, 'ax4\ngs[-1, 0]', **opts) 
ax5 = fig.add_subplot(gs[ - 1, - 2]) 
ax5.text(0.5, 0.5, 'ax5\ngs[-1, -2]', **opts)

plt.show()

fig = plt.figure(constrained_layout=True)
w = [2, 3, 1.5]
h = [1, 3, 2]
spec = fig.add_gridspec(ncols=3, nrows=3, width_ratios=w,
                        height_ratios=h)
for row in range(3):
    for col in range(3):
        label = 'Width: {}\nHeight: {}'.format(w[col], h[row])
        ax = fig.add_subplot(spec[row, col])
        ax.annotate(label, (0.1, 0.5),
                    xycoords='axes fraction', va='center')

plt.show()

gs_kw = dict(width_ratios=w, height_ratios=h)
fig, axs = plt.subplots(ncols=3, nrows=3, constrained_layout=True,
                        gridspec_kw=gs_kw)
for r, row in enumerate(axs):
    for c, ax in enumerate(row):
        lbl = 'Width: {}\nHeight: {}'.format(w[c], h[r])
        ax.annotate(lbl, (0.1, 0.5),
                    xycoords='axes fraction', va='center')

plt.show()


fig, axs = plt.subplots(ncols=3, nrows=3)
gs = axs[0, 0].get_gridspec()

for ax in axs[1:, -1]:
    ax.remove()

axcom = fig.add_subplot(gs[1:, -1])
axcom.annotate('Combined Axes \nGridSpec[1:, -1]', (0.1, 0.5),
               xycoords='axes fraction', va='center')

fig.tight_layout()

plt.show()
