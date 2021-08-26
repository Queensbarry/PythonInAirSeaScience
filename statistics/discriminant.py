"""
Discriminant Analysis
"""
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as Qda
from ._typing import array_like, NoReturn
from .stepwise import StepWise


class QuadraticDiscriminantAnalysis:
    def __init__(self, x: array_like, y: array_like) -> NoReturn:
        self.qda = Qda(store_covariance=True)
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.pvalue = None

    def __call__(self) -> callable:
        self.qda.fit(self.x, self.y)
        # TODO: 显著性检验
        return self

    def predict(self, x: array_like):
        x = np.array(x)

        return self.qda.predict(x)


class MahalanobisDistanceDiscriminant:
    """
    Distance judgement by Magalanobis.
    """
    def __init__(self, a: array_like, b: array_like) -> NoReturn:
        """
        :param a: collections
        :param b: collections
        """
        self.a = np.asarray(a)
        self.b = np.asarray(b)

        self._category = None

    def __call__(self, a_obs: float, b_obs: float):
        x = np.array([a_obs, b_obs]).reshape(-1, 1)

        mu_1 = self.a.mean(axis=0).reshape(-1, 1)
        mu_2 = self.b.mean(axis=0).reshape(-1, 1)

        sigma_1 = np.cov(self.a, rowvar=False)
        sigma_2 = np.cov(self.b, rowvar=False)

        if (sigma_1 == sigma_2).all():
            mu_overline = (mu_1 + mu_2) / 2
            omega = (x - mu_overline).T * np.linalg.inv(sigma_1) * (mu_1 - mu_2)
        else:
            omega = np.dot((x - mu_2).T, np.dot(np.linalg.inv(sigma_2), (x - mu_2))) - \
                    np.dot((x - mu_1).T, np.dot(np.linalg.inv(sigma_1), (x - mu_1)))

        self._category = 0 if omega >= 0 else 1

        return self

    @property
    def category(self) -> int:
        if self._category is None:
            raise RuntimeError('Please execute this module first.')

        return self._category


class StepwiseDiscriminant(StepWise):
    def __init__(self,
                 x: array_like or pd.DataFrame, y: array_like or pd.Series,
                 criteria, processing='dummy_drop_first'):
        super(StepwiseDiscriminant, self).__init__(x, y, model='logistic', criteria=criteria, processing=processing)
