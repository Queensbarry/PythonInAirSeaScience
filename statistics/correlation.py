import numpy as np
import scipy as sp
import scipy.stats
from collections import namedtuple
from ._typing import array_like, array_check, array_cross_check, int_check, number_check, Number


class Statistics:
    """
    Statistics base class, it contain t-test and f-test function.
    """

    def __init__(self) -> None:
        self._statistics = None

    def __call__(self, *args, **kwargs):
        pass

    @property
    def statistics(self) -> float:
        if self._statistics is None:
            self.__call__()

        return self._statistics

    @statistics.setter
    def statistics(self, value: float):
        self._statistics = value

    def t_test(self, alpha: float) -> bool:
        pass

    def f_test(self, alpha: float) -> bool:
        pass


class Pearsonr(Statistics):
    """
    Pearsonr correlation coefficient and t-test.
    """

    PearsonrResult = namedtuple('PearsonrResult', ('correlation', 'pvalue'))

    def __init__(self, x: array_like, y: array_like) -> None:
        """
        :param x: array_like
        :param y: array_like
        """

        super(Pearsonr, self).__init__()
        self.x, self.y = array_cross_check(x, y, 1)

    def __call__(self):
        """
        Calculate Pearsonr correlation coefficient

        :return: PearsonrResult
            corelation and pvalue
        """

        pearsonr_result = self.PearsonrResult(*sp.stats.pearsonr(self.x, self.y))
        self.statistics = pearsonr_result.correlation

        return pearsonr_result

    def t_test(self, alpha: float) -> bool:
        """
        t-test

        :param alpha: float
            significant value
        :return: bool
            passing test label
        """
        alpha = number_check(alpha, 0, 1)
        if self.statistics is None:
            # If self.statistics is None mean that this module have not been run.
            # Run this module and get the self.statistics
            self.__call__()

        n = self.x.size

        # t-statistics
        t = np.sqrt(n - 2) * self.statistics / np.sqrt(1 - self.statistics ** 2)

        return np.abs(t) > sp.stats.t.isf(alpha, n - 2)


class Spearman(Statistics):
    """
    Spearman correlation coefficient and t-test.
    """

    SpearmanResult = namedtuple('SpearmanResult', ('correlation', 'pvalue'))

    def __init__(self, x: array_like, y: array_like) -> None:
        """
        :param x: array_like
        :param y: array_like
        """
        super(Spearman, self).__init__()
        self.x, self.y = array_cross_check(x, y)

    def __call__(self):
        """
        Calculate Spearman correlation coefficient

        :return: SpearmanResult
            correlation and pvalue
        """

        spearman_result = self.SpearmanResult(*sp.stats.spearmanr(self.x, self.y))
        self.statistics = spearman_result.correlation

        return spearman_result

    def t_test(self, alpha: float) -> bool:
        """
        t-test

        :param alpha: float
            significant value
        :return: bool
            passing test label
        """
        alpha = number_check(alpha, 0, 1)
        if self.statistics is None:
            # If self.statistics is None mean that this module have not been run.
            # Run this module and get the self.statistics
            self.__call__()

        n = self.x.size

        # t-statistics
        t = self.statistics * np.sqrt(n - 2) / np.sqrt(1 - self.statistics ** 2)

        return np.abs(t) > sp.stats.t.isf(alpha, n - 2)


class ThreeVariable(Statistics):
    """
    Three partial correlation coefficient and t-test
    """

    def __init__(self, x_1: array_like, x_2: array_like, x_3: array_like) -> None:
        """
        :param x_1: array_like
        :param x_2: array_like
        :param x_3: array_like
        """
        super(ThreeVariable, self).__init__()

        array_cross_check(x_1, x_2)
        array_cross_check(x_1, x_3)
        array_cross_check(x_2, x_3)

        self.x_1 = x_1
        self.x_2 = x_2
        self.x_3 = x_3

    def __call__(self):
        """
        Calculate three partial correlation coefficient.

        :return: class
            self
        """
        r_ab = sp.stats.pearsonr(self.x_1, self.x_2)[0]
        r_ac = sp.stats.pearsonr(self.x_1, self.x_3)[0]
        r_bc = sp.stats.pearsonr(self.x_2, self.x_3)[0]

        self.statistics = (r_ab - r_ac * r_bc) / (((1 - r_ac ** 2) ** 0.5) * ((1 - r_bc ** 2) ** 0.5))

        return self

    def t_test(self, alpha: float) -> bool:
        """
        t-test

        :param alpha: float
            significant value
        :return: bool
            passing test label
        """
        alpha = number_check(alpha, 0, 1)
        if self.statistics is None:
            # If self.statistics is None mean that this module have not been run.
            # Run this module and get the self.statistics
            self.__call__()

        n = self.x_1.size
        t = np.sqrt(n - 5) * self.statistics / np.sqrt(1 - self.statistics ** 2)

        return np.abs(t) > sp.stats.t.isf(alpha, n - 2)


class DelayAuto(Statistics):
    """
    Auto correlation coefficient and t-test
    """

    def __init__(self, a: array_like):
        """
        :param a: array_like
        """
        super(DelayAuto, self).__init__()

        self.a = array_check(a, 1)
        self.tau = None

    def __call__(self, tau: int):
        """
        Calculate auto correlation.

        :param tau: int
            delay length
        :return: class
            self
        """
        self.tau = int_check(tau, 0)

        n = self.a.size
        _sum = 0
        for i in range(n - self.tau):
            _sum += (self.a[i] - self.a.mean()) * (self.a[i + self.tau] - self.a.mean()) / self.a.var()
        self.statistics = _sum / (n - self.tau)

        return self

    def t_test(self, alpha: float) -> bool:
        """
        t-test

        :param alpha: float
            significant value
        :return: bool
            passing test label
        """
        alpha = number_check(alpha, 0, 1)
        if self.statistics is None:
            raise RuntimeError('Please execute this module first.')

        n = self.a.size
        t = np.sqrt(n - 2 - self.tau) * self._statistics / np.sqrt(1 - self._statistics ** 2)

        return np.abs(t) > sp.stats.t.isf(alpha / 2, n - 2 - self.tau)


class DelayCross(Statistics):
    """
    Two variable delay cross correlation.
    """

    def __init__(self, x: array_like, y: array_like):
        """
        :param x: array_like
        :param y: array_like
        """
        super(DelayCross, self).__init__()
        self.x, self.y = array_cross_check(x, y, 1)
        self.tau = None

    def __call__(self, tau: int):
        """
        Calculate delay cross correlation.

        :param tau: int
            delay length
        :return: class
            self
        """
        self.tau = tau
        n = self.x.size
        _sum = 0
        for i in range(n - self.tau):
            _sum += ((self.x[i] - self.x.mean()) / self.x.std()) * \
                    ((self.y[i + self.tau] - self.y.mean()) / self.y.std())
        self.statistics = _sum / (n - self.tau)

        return self

    def t_test(self, alpha: float) -> bool:
        """
        t-test

        :param alpha: float
            significant value
        :return: bool
            passing test label
        """
        alpha = number_check(alpha, 0, 1)
        if self.statistics is None:
            raise RuntimeError('Please execute this module first.')

        n = self.x.size
        t = np.sqrt(n - 2 - self.tau) * self.statistics / np.sqrt(1 - self.statistics ** 2)

        return t > sp.stats.t.isf(alpha / 2, n - 2 - self.tau)


def clime_depature(clime: array_like, forecast: array_like, observation: array_like) -> Number:
    """
    Calculate clime depature

    :param clime: array_like
        1-D or 2-D array
    :param forecast: array_like
        1-D or 2-D array
    :param observation: array_like
        1-D or 2-D array
    :math:
        TODO
    :return: Number
    """
    # TODO: array check must add into here.

    clime = np.asarray(clime)
    forecast = np.asarray(forecast)
    observation = np.asarray(observation)

    n = clime.size

    mfc = (forecast - clime).sum() / n
    moc = (observation - clime).sum() / n

    r_ano = ((forecast - clime - mfc) * (observation - clime - moc)).sum() / np.power(
        np.power(forecast - clime - mfc, 2).sum() *
        np.power(observation - clime - moc, 2).sum(),
        1 / 2)

    return r_ano
