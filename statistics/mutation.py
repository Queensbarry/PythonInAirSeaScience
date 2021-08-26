import numpy as np
import scipy as sp
import scipy.stats
from ._typing import array_check, array_like, int_check, number_check


class Mutation:
    """
    Mutation detection base class.
    """
    def __init__(self) -> None:
        self._statistics = None

    def testing(self):
        raise NotImplementedError

    def t_test(self, alpha: float) -> np.ndarray:
        raise NotImplemented('This module is not use T test.')

    def f_test(self, alpha: float) -> np.ndarray:
        raise NotImplemented('This module is not use F test.')

    def norm_test(self, alpha: float) -> np.ndarray:
        raise NotImplemented('This module is not use norm test.')

    def chi2_test(self, alpha: float) -> np.ndarray:
        raise NotImplemented('This module is not use chi2 test.')

    @property
    def statistics(self) -> None or float:
        return self._statistics


class SlideT(Mutation):
    """
    Slide T method.
    """
    def __init__(self, a: array_like, n_1: int, n_2: int) -> None:
        """
        :param a: array_like
            1-D array
        :param n_1: int
            left testing length
        :param n_2: int
            right test length
        """
        super(SlideT, self).__init__()
        self.a = array_check(a, 1)
        self.n_1 = int_check(n_1, 1)
        self.n_2 = int_check(n_2, 1)

    def testing(self):
        """
        Slide and test.
        :return: class
            self
        """
        t = np.zeros(shape=self.a.shape)

        for i, in np.ndindex(self.a.shape):
            if self.n_1 <= i <= self.a.size - self.n_2 - 1:
                x_forward = self.a[i - self.n_1: i]
                x_backward = self.a[i + 1: i + self.n_2 + 1]
                s = np.sqrt(
                    (self.n_1 * np.var(x_forward) + self.n_2 * np.var(x_backward)) /
                    (self.n_1 + self.n_2 - 2)
                )
                t[i] = (np.mean(x_forward) - np.mean(x_backward)) / \
                       (s * np.sqrt(1 / self.n_1 + 1 / self.n_2))

        t[: self.n_1] = 0
        t[-self.n_2:] = 0
        self._statistics = t

        return self

    def t_test(self, alpha: float) -> np.ndarray:
        """
        T-test and find mutation.

        :param alpha: float
            significant level
        :return:
        """
        alpha = number_check(alpha, 0, 1)
        if self._statistics is None:
            self.testing()

        significant = sp.stats.t.isf(alpha / 2, self.n_1 + self.n_2 - 2)

        return np.nonzero(np.abs(self._statistics) >= significant)[0]


class Cramer(Mutation):
    """
    Cramer mutation detection.
    """
    def __init__(self, a: array_like, n: int):
        """
        :param a: array_like
            1-D array
        :param n: int
            testing length
        """
        super(Cramer, self).__init__()

        self.a = array_check(a, 1)
        self.n = int_check(n, 1)

    def testing(self):
        """
        Slide and test.

        :return: class
            self
        """
        t = np.zeros(shape=self.a.shape, dtype=np.float)

        for i, item in enumerate(np.nditer(t[: -self.n + 1], op_flags=['readwrite'])):
            x = self.a[i: i + self.n]
            tau = (x.mean() - self.a.mean()) / self.a.std()
            item[...] = np.sqrt((self.n * (self.a.size - 2)) /
                                (self.a.size - self.n * (1 + tau))) * tau

        t[-self.n:] = 0
        self._statistics = t

        return self

    def t_test(self, alpha: float) -> np.ndarray:
        """
        T-test and find mutation.

        :param alpha: float
            significant level
        :return:
        """
        alpha = number_check(alpha, 0, 1)
        if self._statistics is None:
            self.testing()

        significant = sp.stats.t.isf(alpha / 2, self.n - 2)

        return np.nonzero(np.abs(self._statistics) >= significant)[0]


class Yamamoto(Mutation):
    """
    Yamamoto mutation detection.
    """

    def __init__(self, a: array_like, n_1: int, n_2: int) -> None:
        """
        :param a: array_like
            1-D array
        :param n_1: int
            left testing length
        :param n_2: int
            right test length
        """
        super(Yamamoto, self).__init__()
        self.a = array_check(a, 1)
        self.n_1 = int_check(n_1, 1)
        self.n_2 = int_check(n_2, 1)

    def testing(self):
        """
        Slide and test.

        :return: class
            self
        """
        r = np.zeros(shape=self.a.shape)

        for i, item in enumerate(np.nditer(r, op_flags=['readwrite'])):
            if self.n_1 <= i <= self.a.size - self.n_2 - 1:
                x_forward = self.a[i - self.n_1: i]
                x_backward = self.a[i + 1: i + self.n_2 + 1]
                item[...] = np.abs(x_forward.mean() - x_backward.mean()) / (np.std(x_forward) + np.std(x_backward))

        r[: self.n_1] = 0
        r[-self.n_2:] = 0

        self._statistics = r

    @property
    def mutation(self) -> np.ndarray:
        """
        Mutation simple

        :return: np.ndarray
        """
        if self._statistics is None:
            self.testing()

        return np.where((self._statistics >= 1) & (self._statistics < 2))[0]

    @property
    def strong_mutation(self) -> np.ndarray:
        """
        Strong mutation.

        :return: np.ndarray
        """
        if self._statistics is None:
            self.testing()

        return np.where(self._statistics >= 2)[0]


class MannKendall(Mutation):
    """
    Mann Kendall mutation detection.
    """
    def __init__(self, a: array_like) -> None:
        """
        :param a: array_like
            1-D array
        """
        super(MannKendall, self).__init__()
        self.a = array_check(a, 1)

        self._uf = None
        self._ub = None

    def testing(self):
        """
        Testing.

        :return: class
            self
        """
        self._uf = self._time_seq(self.a)
        self._ub = -np.flipud(self._time_seq(np.flipud(self.a)))

        return self

    @property
    def uf(self) -> np.ndarray:
        if self._uf is None:
            self.testing()

        return self._uf

    @property
    def ub(self) -> np.ndarray:
        if self._ub is None:
            self.testing()

        return self._ub

    @property
    def intersection(self) -> np.ndarray:
        """
        Intersection index

        :return: np.ndarray
        """
        if self._uf is None:
            self.testing()

        diff = self._uf - self._ub

        return np.where(np.multiply(diff[:-1], diff[1:]) < 0)[0]

    def norm_test(self, alpha: float) -> bool:
        """
        Test up and down tendency is significant or not.

        :param alpha: float
            significant level
        :return: bool
        """
        return self.uf[self.intersection] > sp.stats.norm.isf(alpha / 2)

    @staticmethod
    def _time_seq(a: np.ndarray) -> np.ndarray:
        """
        NOTICE: INTERNAL FUNCTION, DO NOT CALL OUTSIDE.
        Calculate time rank sequence in Mann Kendall method.

        :param a: array_like
            1-D array
        :return: np.ndarray
        """
        s = np.empty(shape=a.shape)
        s_k_mean = np.empty(shape=a.shape)
        s_k_var = np.empty(shape=a.shape)
        for i, item in enumerate(np.nditer(s, op_flags=['readwrite']), 1):
            if i == 1:
                continue
            item[...] = np.count_nonzero(a[: i - 1] < a[i - 1])
            s_k_mean[i - 1] = i * (i + 1) / 4
            s_k_var[i - 1] = i * (i - 1) * (2 * i + 5) / 72

        s = s.cumsum()
        s_k = (s - s_k_mean) / np.sqrt(s_k_var)
        s_k[0] = 0

        return s_k


class Pettitt(Mutation):
    """
    Pettitt mutation detection.
    """
    def __init__(self, a: array_like) -> None:
        """
        :param a: array_like
            1-D array
        """
        super(Pettitt, self).__init__()
        self.a = array_check(a, 1)

        # time rank sequence
        self._s = None

    def testing(self):
        """
        Testing

        :return: class
            self
        """
        s = np.empty(shape=self.a.shape)
        for i, item in enumerate(np.nditer(s, op_flags=['readwrite'])):
            bigger = np.count_nonzero(self.a[: i] < self.a[i])
            smaller = np.count_nonzero(self.a[: i] > self.a[i])
            item[...] = bigger - smaller
        self._s = s

    @property
    def mutation(self) -> np.ndarray:

        if self._s is None:
            self.testing()
        _max = np.max(np.abs(self._s))

        return np.where(np.abs(self._s) == _max)[0]

    @property
    def siginificant(self) -> np.ndarray:

        if self._s is None:
            self.testing()
        _max = np.max(np.abs(self._s))

        return 2 * np.exp(-6 * _max**2 * (self.a.size**3 + self.a.size**2)) <= 0.5
