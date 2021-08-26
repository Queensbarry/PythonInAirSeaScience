import numpy as np
import scipy as sp
import scipy.stats

from ._typing import array_check, array_cross_check, array_like, int_check, number_check, Number, Tuple
from .regression import AutoRegression
from .variance import delay_cross_covariance, auto_covariance
from .utils import prepare_empty_container_with_same_size, hanning_smoothing, standardization
from .correlation import DelayAuto


class PowerSpectrum:
    """
    Power sepectrum
    """
    def __init__(self, a: array_like):
        """
        :param a: array_like
            1-D array
        """

        self.a = array_check(a, 1)

        self._n = a.size
        self._s = None
        self._r = None
        self.m = None

    def fit(self, m: int):
        """
        Fit method

        :param m: int
            delay length
        :return: class
            self
        """
        self.m = int_check(m, 0)

        r, s, s_smoothing = prepare_empty_container_with_same_size((m + 1,), 3)

        dac_ = DelayAuto(self.a)
        dac = np.vectorize(lambda x: dac_(x).statistics)
        r = dac(np.arange(m + 1))

        s[0] = (r[0] + r[-1]) / (2 * m) + r[1: -1].sum() / m
        for k in range(1, m):
            _center = 2 * np.multiply(np.cos(k * np.pi * np.arange(1, m) / m), r[1: -1]).sum()
            _right = r[-1] * np.cos(k * np.pi)
            s[k] = (r[0] + _center + _right) / m
        s[-1] = (r[0] + (-1) ** m * r[-1]) / (2 * m) + \
                (np.power(-1, np.arange(1, m)) * r[1: -1]).sum() / m

        self._s = hanning_smoothing(s)
        self._r = r

        return self

    @property
    def s(self) -> np.ndarray:
        """
        Get statistics sequence.

        :return: np.ndarray
        """
        if self._s is None:
            raise RuntimeError('Please execute this module first.')

        return self._s

    def t(self, alpha: float) -> np.ndarray:
        """
        Get cycle while passing chi2 test.
        :param alpha:
        :return:
        """
        alpha = number_check(alpha, 0, 1)
        _t = 2 * self.m / np.arange(1, self.m + 1)

        return np.squeeze(_t[self.chi2_test(alpha)[1:]])

    def chi2_test(self, alpha: float, positive: Number = 0.01) -> np.ndarray:
        """
        chi2 test

        :param alpha: alpha
            significant value
        :param positive: Number
            identify red or white noise
        :return: np.ndarray
            test result
        """
        alpha = number_check(alpha, 0, 1)
        if self._s is None:
            raise RuntimeError('Please execute this module first.')

        s0 = np.empty(self._s.shape)

        _s = (self._s[0] + self._s[-1]) / (2 * self.m) + self._s[1: -1].sum() / self.m
        if self._r[1] > positive:
            k = np.arange(0, self.m + 1)
            s0 = _s * ((1 - self._r[1] ** 2) / (1 + self._r[1] ** 2 - 2 * self._r[1] * np.cos(np.pi * k / self.m)))
        else:
            s0[:] = _s

        v = (2 * self._n - self.m / 2) / self.m

        sok = s0 * (sp.stats.chi2.isf(alpha, v) / v)

        return self._s > sok


class MaximumEntropySpectralEstimation:
    def __init__(self, a: array_like):
        a = np.asarray(a)
        if a.ndim != 1:
            raise ValueError('Input array must be 1-D.')

        self.a = a

    def fit(self):

        for k in range(1, self.a.size):
            ar = AutoRegression(self.a, k)
            ar()
            print(ar.result.sigma2, ar.result.params)
            # print()


class CrossSpectrum:
    """
    Cross spectrum.
    """
    def __init__(self, a: array_like, b: array_like, tau: int) -> None:
        """
        :param a: array_like
            1-D array
        :param b: array_like
            1-D array
        :param tau: int
            delay length
        """
        self.a, self.b = array_cross_check(a, b, 1)
        self.m = int_check(tau, 0)

        self._cospectrum = None
        self._quadrature = None
        self._amplitude = None
        self._phase = None
        self._condensation = None

    def fit(self):
        """
        Calculate all variable.

        :return: class
            self
        """
        r12, r21, p, q = prepare_empty_container_with_same_size((self.m + 1,), 4)
        for index, item in enumerate(np.nditer([r12, r21], [], ['writeonly', 'writeonly'])):
            item[0][...] = delay_cross_covariance(self.a, self.b, index)
            item[1][...] = delay_cross_covariance(self.b, self.a, index)

        for k in range(self.m + 1):
            sum_p = 0
            sum_q = 0
            for j in range(1, self.m):
                sum_p = (r12[j] + r21[j]) * np.cos(np.pi * k * j / self.m)
                sum_q = (r12[j] - r21[j]) * np.sin(np.pi * k * j / self.m)
            p[k] = (r12[0] + sum_p + r12[-1] * np.cos(k * np.pi)) / self.m
            q[k] = sum_q / self.m

        self._cospectrum = p
        self._quadrature = q

        return self

    def _smooth_p_q(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        NOTICE: INTERNAL FUNCTION, DO NOT CALL OUTSIDE.

        :return: smoothing result
        """
        p12 = hanning_smoothing(self._cospectrum)
        q12 = hanning_smoothing(self._quadrature)

        p11 = PowerSpectrum(self.a).fit(self.m).s
        p22 = PowerSpectrum(self.b).fit(self.m).s

        return p11, p22, p12, q12

    def t(self, alpha: float) -> np.ndarray:
        """
        Get cycle.

        :param alpha: float
            significant level
        :return: np.ndarray
        """
        _t = 2 * self.m / np.arange(1, self.m + 1)

        return np.squeeze(_t[self.f_test(alpha)[1:]])

    def delay_time_length_spectrum(self, alpha: float):
        _l = (self.m * self.phase[1:]) / (np.pi * np.arange(1, self.m + 1))

        return np.squeeze(_l[self.f_test(alpha)[1:]])

    def f_test(self, alpha) -> np.ndarray:
        if self._cospectrum is None:
            raise RuntimeError('Please execute this module first.')

        v = (2 * self.a.size - (self.m - 1) / 2) / (self.m - 1)

        statistics = (v - 1) * self.condensation / (1 - self.condensation)

        return statistics > sp.stats.f.isf(alpha, 2, 2 * (v - 1))

    @property
    def amplitude(self):
        if self._cospectrum is None:
            raise RuntimeError('Please execute this module first.')

        _, _, p12, q12 = self._smooth_p_q()

        return np.sqrt(np.power(p12, 2) + np.power(q12, 2))

    @property
    def phase(self):
        if self._cospectrum is None:
            raise RuntimeError('Please execute this module first.')

        _, _, p12, q12 = self._smooth_p_q()

        return np.arctan(q12, p12)

    @property
    def condensation(self):
        if self._cospectrum is None:
            raise RuntimeError('Please execute this module first.')

        p11, p22, p12, q12 = self._smooth_p_q()

        return (np.power(p12, 2) + np.power(q12, 2)) / np.multiply(p11, p22)


class SingularSpectrumAnalysis:

    standardization = staticmethod(standardization)

    def __init__(self, x: array_like, m: int, standard: bool = False):

        self.x = array_check(x, 1)

        if not standard:
            x = standardization(x)

        nt = x.size
        n = nt - m + 1
        _x = np.empty(shape=(m, n))
        for i in range(m):
            _x[i] = x[i: i + n]

        self.x = _x

        s_ = np.empty(shape=(m,))
        for j in np.arange(m):
            s_[j] = auto_covariance(x, j)

        s = np.empty(shape=(m, m))
        s[0] = s_
        s[:, 0] = s_
        for i, j in np.ndindex(s.shape):
            if (i == 0) or (j == 0):
                continue
            s[i, j] = s[i - 1, j - 1]

        lambda_, hai = np.linalg.eig(s)

        t_pc = np.empty(shape=(m, n))
        for k, i in np.ndindex(t_pc.shape):
            t_pc[k, i] = (x[i + np.arange(m)] * hai[k]).sum()

        sort_index = np.argsort(-lambda_)
        lambda_ = lambda_[sort_index]
        t_pc = t_pc[:, sort_index]

        # np.savetxt('lambda.txt', lambda_, fmt='%.3f')
        # np.savetxt('./tpc.txt', t_pc, fmt='%.3f')
        # np.savetxt('./teof.txt', hai, fmt='%.3f')

        var_contribution = lambda_ / (lambda_ ** 2).sum()

        nd = (nt / m) - 1
        delta_lambda = np.sqrt(2 / nd) * lambda_

        _condition = np.diff(lambda_) <= np.minimum(delta_lambda[: -1], delta_lambda[1:])

        for i in np.ndindex(_condition.shape):
            cross_covariance = np.abs(delay_cross_covariance(
                np.squeeze(hai[:, i[0] + 1].reshape(1, -1)),
                np.squeeze(hai[:, i].reshape(1, -1)),
                np.arange(m)[i]
            ))
            if cross_covariance >= 0.90:
                print(np.arange(m)[i], cross_covariance)
