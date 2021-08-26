import numpy as np
from ._typing import array_like, Callable
from .interp import Newton


class Remez:
    def __init__(self, f: Callable, deg: int, a: float, b: float, eps=1e-3):
        self.f = f
        self.n = deg

        if a >= b:
            raise ValueError('a must smaller than b.')

        self.a = a
        self.b = b
        self.eps = eps

    def fit(self):
        k = np.arange(self.n + 1)
        xk = (self.b + self.a + (self.b - self.a) * np.cos((self.n - k) * np.pi / self.n)) / 2

        newton = Newton(k, self.f(k))
        # yk = newton(xk)

        _x = np.arange(self.a, self.b, self.eps)
        _max_index = np.argmax(np.abs(self.f(_x) - newton(_x)))

        _max_value = _x[_max_index]
        print(self.a, _max_value, xk[0])
        if self.a <= _max_value < xk[0]:
            print(1)
        elif xk[-1] < _max_value <= self.b:
            print(2)
        else:
            print(3)

    def _mu(self, x: np.array, newton_y: np.array):
        newton = Newton(x, newton_y)

        return np.abs(self.f(x) - newton(x)).max()
