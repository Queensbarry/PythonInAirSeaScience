import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.cm.jet
norm = mpl.colors.Normalize(vmin=5, vmax=10)

cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
cb.set_label('Data Range')
fig.show()


fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.colors.ListedColormap(
            ['blue', 'cyan', 'green', 'orange', 'red'])
cmap.set_over('0.8')
cmap.set_under('0.2')

bounds = [0, 1, 3, 6, 7, 9]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                boundaries=[-4] + bounds + [14],
                                extend='both',
                                ticks=bounds,
                                spacing='proportional',
                                orientation='horizontal')
cb.set_label('Discrete Intervals')
fig.show()

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.colors.ListedColormap(['royalblue', 'cyan',
                                  'yellow', 'orange'])
cmap.set_over('red')
cmap.set_under('blue')

bounds = [-1.0, -0.5, 0.0, 0.5, 1.0]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                boundaries=[-10] + bounds + [10],
                                extend='both',
                                extendfrac='auto',
                                ticks=bounds,
                                spacing='uniform',
                                orientation='horizontal')
cb.set_label('Custom Extension Lengths')
fig.show()

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.colors.ListedColormap(
            ['blue', 'cyan', 'green', 'orange', 'red'])
cmap.set_over('0.8')
cmap.set_under('0.2')

bounds = [0, 1, 3, 6, 7, 9]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                boundaries=[-4] + bounds + [14],
                                extend='both',
                                extendfrac='auto',
                                ticks=bounds,
                                spacing='proportional',
                                orientation='horizontal')
cb.set_label('Extensions for Different Intervals')
fig.show()

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.colors.ListedColormap(
            ['blue', 'cyan', 'green', 'orange', 'red'])
cmap.set_over('0.8')
cmap.set_under('0.2')

bounds = [0, 1, 3, 6, 7, 9]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                boundaries=[-4] + bounds + [14],
                                extend='both',
                                extendfrac=0.1,
                                ticks=bounds,
                                spacing='proportional',
                                orientation='horizontal')
cb.set_label('Equal Extension Lengths')
fig.show()

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.cm.jet
cmap.set_over('0.8')
cmap.set_under('black')
norm = mpl.colors.Normalize(vmin=5, vmax=10)

cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                extend='both',
                                extendfrac=(0.1, 0.1),
                                orientation='horizontal')
cb.set_label('Data Range')
fig.show()
