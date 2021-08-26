import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import statsmodels.api as sm
from dataclasses import dataclass
from statsmodels.api import OLS
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import ARMA
from ._typing import array_like, array_check, int_check, number_check
from .correlation import DelayAuto
from .stepwise import StepWise


@dataclass
class LinearRegressionParams:
    """
    Linear regression params.
    """
    Coef: np.array
    Intercept: float
    pvalues: float
    F_statistics: float
    R_square: float


class LinearRegression:
    """
    Linear regression for single variable or multi-variable.
    """
    def __init__(self, x: array_like, y: array_like) -> None:
        """
        :param x: array_like
            N-D array with (n-sample, n-feature)
        :param y: array_like
        """
        self.x = array_check(x)
        self.y = array_check(y)

        self._coef = None
        self._intercept = None
        self._pvalues = None
        self._f_statistics = None
        self._r_squared = None

        self.__m = OLS(self.y, sm.add_constant(self.x))

    def fit(self) -> LinearRegressionParams:
        """
        Fit the self.x and self.y, then get fitting params.

        :return: class
            LinearRegressionParams
        """
        result = self.__m.fit()
        self._intercept, *self._coef = result.params
        self._pvalues = result.f_pvalue
        self._f_statistics = result.fvalue
        self._r_squared = result.rsquared

        return LinearRegressionParams(
            Coef=self._coef,
            Intercept=self._intercept,
            pvalues=self._pvalues,
            F_statistics=self._f_statistics,
            R_square=self._r_squared
        )

    def predict(self, x: array_like) -> np.ndarray:
        """
        predict new values

        :return: np.ndarray
            new predict values
        """
        x = array_check(x)

        if x.ndim == 1:
            result = np.squeeze(self._coef * x + self._intercept)
        else:
            result = np.squeeze(np.multiply(self._coef, x).sum(axis=1) + self._intercept)

        return result

    def f_test(self, alpha: float) -> bool:
        """
        F-test

        :param alpha: float
            significant value
        :return: bool
            passing f-test label
        """
        alpha = number_check(alpha, 0, 1)
        if self.x.ndim == 1:
            n = self.x.size
            result = self._f_statistics > sp.stats.f.isf(alpha, 1, n - 2)
        else:
            n, p = self.x.shape
            result = self._f_statistics > sp.stats.f.isf(alpha, p, n - p - 1)

        return result


class StepwiseRegression(StepWise):
    """
    Step wise Regression
    """
    def __init__(self,
                 x: array_like or pd.DataFrame, y: array_like or pd.Series,
                 criteria, processing='dummy_drop_first'):
        super(StepwiseRegression, self).__init__(x, y, model='regression', criteria=criteria, processing=processing)


class AutoRegression:
    """
    Auto regression
    """
    def __init__(self, x: array_like, tau: int) -> None:
        """
        :param x: array_like
            1-D array
        :param tau: int
            delay length
        """
        self.x = array_check(x, 1)
        self.tau = int_check(tau, 0)

        self.__m = AutoReg(x, tau, hold_back=tau)
        self._result = None

    def fit(self):
        """
        Fit the model

        :return: class
            self
        """
        self._result = self.__m.fit()

        return self

    @property
    def r(self) -> float:
        """
        R result

        :return: float
        """
        if self._result is None:
            raise RuntimeError('Please execute this module first.')

        ac = DelayAuto(self.x)
        r_k = np.empty(shape=(self.tau,))
        for i in range(self.tau):
            r_k[i] = ac(i + 1).statistics

        return np.multiply(r_k, self._result.params[1:]).sum()

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError('Please execute this module first.')

        return self._result


class AutoRegressionMovingAverage:
    """
    Auto Regression Moving Average
    """
    def __init__(self, x: array_like, p: int, q: int) -> None:
        self.x = array_check(x, 1)
        self.p = int_check(p, 0)
        self.q = int_check(q, 0)

        self.__m = ARMA(x, (p, q))
        self.result = None

    def fit(self):
        self.result = self.__m.fit(disp=False)

        return self

    @property
    def aic(self):
        if self.result is None:
            raise RuntimeError('Please execute this module first.')

        return self.result.aic

    @property
    def bic(self):
        if self.result is None:
            raise RuntimeError('Please execute this module first.')

        return self.result.bic
