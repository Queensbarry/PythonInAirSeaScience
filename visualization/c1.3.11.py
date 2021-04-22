import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect(1)

x1 = -1 + np.random.randn(100)
y1 = -1 + np.random.randn(100)
x2 = 1. + np.random.randn(100)
y2 = 1. + np.random.randn(100)

ax.scatter(x1, y1, color="r")
ax.scatter(x2, y2, color="g")

bbox1 = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)
ax.text(-2, -2, "Sample A", ha="center", va="center", size=20,
        weight=600, color="maroon", bbox=bbox1)
ax.text(2, 2, "Sample B", ha="center", va="center", size=20,
        weight=600, color="darkgreen", bbox=bbox1)


bbox2 = dict(boxstyle="rarrow", fc=(0.5, 1.0, 1.0), ec="b", lw=3)
t = ax.text(0, 0, "Direction", ha="center", va="center", rotation=45,
            size=15, bbox=bbox2)

bb = t.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.6)

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

plt.show()
