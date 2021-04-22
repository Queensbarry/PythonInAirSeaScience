import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
fig.suptitle('Bold Figure Suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Axes Title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})

ax.text(2, 6, r'mass-energy equation : $E=mc^2$', fontsize=15)

ax.text(3, 2, 'unicode: Institut für Festkörperphysik')

ax.text(0.95, 0.01, 'colored text in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)


ax.plot([2], [1], 'o', color='r', ms=12)
ax.annotate('annotation', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.09))

ax.axis([0, 10, 0, 10])

plt.show()
