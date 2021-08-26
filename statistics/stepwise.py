import inspect
import numpy as np
import pandas as pd
from statsmodels.api import add_constant, Logit, OLS
from ._typing import array_like


class StepWise:

    INTERCEPT = ['intercept']

    def __init__(self,
                 x: array_like or pd.DataFrame, y: array_like or pd.Series,
                 model, criteria, processing='dummy_drop_first'):
        """
        :param x: x with (n_sample, n_feature)
        :param y: 1-D array or Series
        :param model: {'regression' or 'logistic'}
            'linear': Step-wise regression
            'logistic': Step-wise discriminant
        :param criteria: {'aic', 'bic', 'r2', 'r2adj'}
            'aic' refers Akaike information criterion
            'bic' refers Bayesian information criterion
            'r2' refers R-squared (Only works on linear model)
            'r2adj' refers Adjusted R-squared (Only works on linear model)
        :param processing: {'drop', 'dummy' or 'dummy_drop_first'}
            'drop' drops varchar features
            'dummy' creates dummies for all levels of all varchars
            'dummy_drop_first' creates dummies for all levels of all varchars, and drops first levels\n
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x, columns=range(x.shape[1]))
        if not isinstance(y, pd.DataFrame):
            y = pd.Series(y)

        if model not in ['regression', 'logistic']:
            raise ValueError('Params model must be one of \'regression\' or \'logistic\'.')

        if criteria not in ['aic', 'bic', 'r2', 'r2adj']:
            raise ValueError('Params criteria must be one of \'aic\', \'bic\', \'r2\' or \'adj_r2\'.')

        if (model == 'logistic') and (criteria in ['r2', 'r2adj']):
            raise ValueError(f'{criteria} only allow uses in linear model.')

        if processing not in ['drop', 'dummy', 'dummy_drop_first']:
            raise ValueError('Params processing must be one of \'drop\', \'dummy\', \'dummy_drop_first\'')
        self.processing = processing
        self.x = self._var_char_processing(x, processing)
        self.y = y

        self.model_type = model
        self.model = None

        self.criteria_standard = criteria
        self.criteria = None

        self._selected = list()
        self._eliminated = list()

    def forward(self, alpha: float):
        """
        :param alpha: significance level
        """
        self.model = self._model(self.x[self.INTERCEPT], self.y, self.model_type)
        self.criteria = self._model_criteria
        self._selected = self.INTERCEPT.copy()
        _cols = self.x.columns.tolist().copy()
        _cols.remove(self.INTERCEPT[0])

        for i in range(self.x.shape[1]):
            pvalues = pd.DataFrame(columns=['col', 'pvalues'])
            for j in _cols:
                self.model = self._model(
                    self.x[self._selected + [j]], self.y,
                    self.model_type)
                pvalues = pvalues.append(pd.DataFrame(
                    [[j, self.model.pvalues[j]]],
                    columns=['col', 'pvalues']
                ), ignore_index=True)

            pvalues = pvalues.sort_values(by=['pvalues']).reset_index(drop=True)
            pvalues = pvalues[pvalues.pvalues <= alpha]
            if pvalues.shape[0] <= 0:
                continue

            self.model = self._model(
                self.x[self._selected + [pvalues['col'][0]]],
                self.y, self.model_type)

            if self._criteria_execute(inspect.stack()[0][3]):
                self._selected.append(pvalues['col'][0])
                _cols.remove(pvalues['col'][0])
                self.criteria = self._model_criteria
            else:
                break

        self.model = self._model(self.x[self._selected], self.y, self.model_type)
        self.criteria = self._model_criteria

        return self

    def backward(self, alpha: float):
        # TODO: eliminated and regained
        _cols = self.x.columns.tolist()
        for i in range(self.x.shape[1]):
            if i != 0:
                self.criteria = self._model_criteria
                self.model = self._model(self.x, self.y, self.model_type)
                if self._criteria_execute(inspect.stack()[0][3]):
                    # regained
                    break
            else:
                self.model = self._model(self.x, self.y, self.model_type)

            _cols = self.x.columns.tolist()
            if self.model.pvalues.max() > alpha:
                for j in _cols:
                    if self.model.pvalues[j] == self.model.pvalues.max():
                        del self.x[j]

        self._selected = _cols

        return self

    def predict(self, x: array_like) -> np.array:
        if self.model is None:
            raise RuntimeError('Please execute this module first.')
        x = np.asarray(x)
        x = add_constant(x)

        return self.model.predict(x)

    def _criteria_execute(self, mode):
        if self.criteria_standard in ['aic', 'bic']:
            condition = self._model_criteria < self.criteria
        else:
            condition = self._model_criteria > self.criteria

        if mode == 'forward':
            condition = condition
        else:
            condition = not condition

        return condition

    @property
    def selected(self) -> list or None:
        if self._selected is None:
            raise RuntimeError('Please execute this module first.')
        return self._selected

    @property
    def aic(self) -> float:
        if self.model is None:
            raise RuntimeError('Please execute this module first.')
        return self.model.aic

    @property
    def bic(self) -> float:
        if self.model is None:
            raise RuntimeError('Please execute this module first.')
        return self.model.bic

    @property
    def r2(self) -> float:
        if self.model_type != 'regression':
            raise AttributeError
        if self.model is None:
            raise RuntimeError('Please execute this module first.')

        return self.model.rsquared

    @property
    def r2adj(self) -> float:
        if self.model_type != 'regression':
            raise AttributeError
        if self.model is None:
            raise RuntimeError('Please execute this module first.')

        return self.model.rsquared_adj

    @property
    def _model_criteria(self) -> float:
        if self.criteria_standard == 'aic':
            result = self.model.aic
        elif self.criteria_standard == 'bic':
            result = self.model.bic
        elif self.criteria_standard == 'r2':
            result = self.model.rsquared
        elif self.criteria_standard == 'r2adj':
            result = self.model.r2adj
        else:
            raise ValueError

        return result

    @staticmethod
    def _model(x: np.array, y: np.array, model: str) -> OLS or Logit:
        """
        :param x: n-D array
        :param y: 1-D array
        :param model: {'linear' or 'logistic'}
        :return: regression model
        """
        if model == 'regression':
            model_ = OLS(y, x).fit()
        else:
            model_ = Logit(y, x).fit()

        return model_

    @staticmethod
    def _var_char_processing(x: pd.DataFrame, processing) -> pd.DataFrame:
        dtypes = x.dtypes
        if processing == 'drop':
            x = x.drop(columns=dtypes[dtypes == np.object].index.tolist())
        elif processing == 'dummy':
            x = pd.get_dummies(x, drop_first=False)
        elif processing == 'dummy_drop_first':
            x = pd.get_dummies(x, drop_first=True)

        x['intercept'] = 1
        cols = x.columns.tolist()
        cols = cols[-1:] + cols[: -1]
        x = x[cols]

        return x
