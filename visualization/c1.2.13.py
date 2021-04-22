import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent,
                       ydescent, width, height, fontsize, trans):
        center = (0.5 * width - 0.5 * xdescent,
                  0.5 * height - 0.5 * ydescent)
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

c = mpatches.Circle((0.5, 0.5), 0.25, facecolor="red",
                    edgecolor="blue", linewidth=2)

plt.legend([c], ["Ellipse handler"],
           handler_map={type(c): HandlerEllipse()})

plt.show()
