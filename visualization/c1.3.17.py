import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0.0, 2.0*np.pi, 100)
s = np.sin(2*t)

plt.plot(t,s)
plt.title(r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',
          fontsize=20)
plt.text(4, -0.6, r'$\sum_{i=0}^\infty x_i$',
         fontsize=20, ha='center')
plt.text(2.3, 0.6, r'$\alpha_i > \beta_i$',
         fontsize=20, ha='center')
plt.xlabel(r'time (s), $\Delta t=0.06$')
plt.ylabel('volts (mV)')
plt.show()
