import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

class AnyObject(object):
    pass

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height,
                                   facecolor='red', edgecolor='blue',
                                   hatch='////', lw=2,
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch

plt.legend([AnyObject()], ['Customized handler'],
           handler_map={AnyObject: AnyObjectHandler()})

plt.show()
