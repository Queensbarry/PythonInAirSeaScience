import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def simple_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ax = plt.subplots(facecolor='0.8')
simple_plot(ax, fontsize=32)
plt.tight_layout()
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = \
        plt.subplots(nrows=2, ncols=2, facecolor='0.8')
simple_plot(ax1)
simple_plot(ax2)
simple_plot(ax3)
simple_plot(ax4)
plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=0.5)
plt.show()

