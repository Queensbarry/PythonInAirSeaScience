import matplotlib.pyplot as plt

fig = plt.figure(figsize=(4, 1.2))
# 普通文本
fig.text(0.1, 0.6, 'plain text:  alpha > beta', size=18)
# 数学文本
fig.text(0.1, 0.25, r'math text:  $\alpha > \beta$', size=18)
plt.show()

fig = plt.figure(figsize=(1, 0.5))
fig.text(0.5, 0.5, r'$\alpha_i > \beta_i$',
         size=16, ha='center', va='center')
plt.show()

fig = plt.figure(figsize=(1, 0.8))
fig.text(0.5, 0.5, r'$\sum_{i=0}^\infty x_i$',
         size=16, ha='center', va='center')
plt.show()

fig = plt.figure(figsize=(1.5, 0.6))
fig.text(0.2, 0.5, r'$\frac{3}{4}$',
        size=16, ha='center', va='center')
fig.text(0.5, 0.5, r'$\binom{3}{4}$',
        size=16, ha='center', va='center')
fig.text(0.8, 0.5, r'$\genfrac{}{}{0}{}{3}{4}$',
        size=16, ha='center', va='center')
plt.show()

fig = plt.figure(figsize=(1.5, 0.6))
fig.text(0.5, 0.5, r'$\frac{5 - \frac{1}{x}}{4}$',
        size=16, ha='center', va='center')
plt.show()

fig = plt.figure(figsize=(1.5, 0.6))
fig.text(0.5, 0.5, r'$(\frac{5 - \frac{1}{x}}{4})$',
        size=16, ha='center', va='center')
plt.show()

fig = plt.figure(figsize=(1.5, 0.6))
fig.text(0.5, 0.5,
        r'$\left(\frac{5 - \frac{1}{x}}{4}\right)$',
        size=16, ha='center', va='center')
plt.show()

fig = plt.figure(figsize=(1.5, 0.5))
fig.text(0.5, 0.5, r'$\sqrt{2}$',
        size=16, ha='center', va='center')
plt.show()

fig = plt.figure(figsize=(1.5, 0.5))
fig.text(0.5, 0.5, r'$\sqrt[3]{x}$',
        size=16, ha='center', va='center')
plt.show()

fig = plt.figure(figsize=(2.5, 0.5))
fig.text(0.5, 0.5,
        r'$s(t) = \mathcal{A}\mathrm{sin}(2 \omega t)$',
        size=16, ha='center', va='center')
plt.show()

fig = plt.figure(figsize=(2.5, 0.5))
fig.text(0.5, 0.5, r'$s(t) = \mathcal{A}\sin(2 \omega t)$',
        size=16, ha='center', va='center')
plt.show()

fig = plt.figure(figsize=(2.5, 0.5))
fig.text(0.5, 0.5, r'$s(t) = \mathcal{A}\/\sin(2 \omega t)$',
        size=16, ha='center', va='center')
plt.show()


