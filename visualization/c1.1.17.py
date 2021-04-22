import matplotlib.pyplot as plt
import numpy as np

N = 5
Means = (21, 35, 30, 32, 27)
ind = np.arange(N)
width = 0.45

with plt.xkcd():
    ax = plt.axes()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.bar(ind, Means, width)
    plt.ylabel('Scores')
    plt.title('Scores by group')
    plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
    plt.yticks(np.arange(0, 41, 10))
    ax.text(2, -6,
        '"The Data So Far" from xkcd by Randall Munroe',
        ha='center')

plt.tight_layout()
plt.show()
