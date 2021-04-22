"""
=========
Pgf Fonts
=========

"""

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    # 使用latex默认的serif字体
    "font.serif": [],
    # 使用指定的sans-serif字体
    "font.sans-serif": ["DejaVu Sans"],
})

plt.figure(figsize=(4.5, 2.5))
plt.plot(range(5))
plt.text(0.5, 3., "serif")
plt.text(0.5, 2., "monospace", family="monospace")
plt.text(2.5, 2., "sans-serif", family="sans-serif")
plt.text(2.5, 1., "comic sans", family="Comic Sans MS")
plt.xlabel("µ is not $\\mu$")
plt.tight_layout(.5)
plt.show()
