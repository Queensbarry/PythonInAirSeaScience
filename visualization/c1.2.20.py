import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

verts = [
   (0., 0.),  # 左下角
   (0., 1.),  # 左上角
   (1., 1.),  # 右上角
   (1., 0.),  # 右下角
   (0., 0.),  # 回到左下角进行封闭
]

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
]

path = Path(verts, codes)

fig, ax = plt.subplots()
patch = patches.PathPatch(path, facecolor='blue',
                          edgecolor='green', lw=2)
ax.add_patch(patch)
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)
plt.show()
