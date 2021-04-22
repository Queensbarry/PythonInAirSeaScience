import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(1, 2)

gs0 = gs[0].subgridspec(2, 3)
gs1 = gs[1].subgridspec(3, 2)

for a in range(2):
    for b in range(3):
        fig.add_subplot(gs0[a, b])
        fig.add_subplot(gs1[b, a])

plt.show()
