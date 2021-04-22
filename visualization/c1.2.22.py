import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

fig = plt.figure(figsize=(5, 1.5))
text = fig.text(0.5, 0.5, 'Normal Effect',
                ha='center', va='center', size=36)
text.set_path_effects([path_effects.Normal()])
plt.show()

fig = plt.figure(figsize=(5, 1.5))
text = fig.text(0.5, 0.5, 'Shadow Effect', color='b',
                ha='center', va='center', size=36, weight=500,
                path_effects=[path_effects.SimplePatchShadow(),
                       path_effects.Normal()])
plt.show()

fig = plt.figure(figsize=(5, 1.5))
text = fig.text(0.5, 0.5, 'Stroke Effect', color='orange',
                ha='center', va='center', size=36)
text.set_path_effects([path_effects.Stroke(linewidth=5, foreground='b'),
                       path_effects.Normal()])
plt.show()

fig = plt.figure(figsize=(5, 1.5))
text = fig.text(0.5, 0.5, 'Hatch Shadow',
                ha='center', va='center', size=36, weight=1000)
text.set_path_effects([path_effects.PathPatchEffect(offset=(4, -4),
                            hatch='xxxx', facecolor='gray'),
                       path_effects.PathPatchEffect(edgecolor='white',
                            linewidth=1.1, facecolor='green')])
plt.show()
