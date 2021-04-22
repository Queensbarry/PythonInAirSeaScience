import matplotlib.pyplot as plt
import numpy as np

image = plt.imread('image/panda.jpg')
fig, ax = plt.subplots()
ax.imshow(image)
ax.axis('off')

plt.show()
