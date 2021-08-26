import numpy as np
import scipy as sp
import scipy.stats
from dataclasses import dataclass
from ._typing import array_check, array_like, array_cross_check, int_check, number_check, NoReturn


@dataclass
class TwoSampleVarDiff:
    """
    Result container for two sample variance diff evaluation.
    """
    statistics: float
    criticality: float = None

    def f_test(self) -> bool:
        """
        F-test result

        :return: bool
            F-test result
        """
        return self.statistics > self.criticality

    def cal_criticality(self, n1: int, n2: int, alpha: float) -> NoReturn:
        """
        Two sample variance diff evaluation's F-test criticality

        :param n1: int
            Numerator degree of freedom
        :param n2: int
            Denominator degree of freedom
        :param alpha: float
            Significant level
        """
        self.criticality = sp.stats.f.cdf(alpha, n1, n2)


def two_sample_var_diff_evaluation(x_1: array_like, x_2: array_like, alpha: float) -> TwoSampleVarDiff:
    """
    Two sample variance diff evaluation

    :param x_1: array_like
        array `x_1`
    :param x_2: array_like
        array `x_2`
    :param alpha: float
        Significant level.
    :math:
        TODO
    :return: class
        TwoSampleVarDiff with statistics, criticality and f-test function.
    """
    x_1, x_2 = array_cross_check(x_1, x_2)
    alpha = number_check(alpha, min=0, max=1)

    f_statistics = x_2.var() / x_1.var()

    tsvd = TwoSampleVarDiff(f_statistics)
    tsvd.cal_criticality(x_1.size - 1, x_2.size - 1, alpha)

    return tsvd


def covariance(x: array_like, y: array_like) -> float:
    """
    Covariance

    :param x: array_like
        sequence `x`
    :param y: array_like
        sequence `y`
    :math:
        TODO
    :return: float
        result
    """
    x, y = array_cross_check(x, y)

    return np.multiply(x - x.mean(), y - y.mean()).sum() / x.size


def auto_covariance(a: array_like, tau: int) -> float:
    """
    Auto covariance

    :param a: array_like
        sequence `a`
    :param tau: int
        delay length
    :math:
        TODO
    :return: float
        result
    """
    a = array_check(a, 1)
    tau = int_check(tau, 0)

    n = a.size
    _sum = 0
    for i in range(n - tau):
        _sum += (a[i] - a.mean()) * (a[i + tau] - a.mean())

    return _sum / (n - tau)


def delay_cross_covariance(x: array_like, y: array_like, tau: int) -> float:
    """
    Delay cross covariance

    :param x: array_like
        sequence `x`
    :param y: array_like
        sequence `y`
    :param tau: int
        delay length
    :math:
        TODO
    :return: int
        result
    """
    x, y = array_cross_check(x, y, 1)
    tau = int_check(tau, 0)

    n = x.size
    _sum = 0
    for i in range(n - tau):
        _sum += ((x[i] - x.mean()) / x.std()) * ((y[i + tau] - y.mean()) / y.std())

    return _sum / (n - tau)
