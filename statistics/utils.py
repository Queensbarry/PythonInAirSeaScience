import numpy as np
from .error import ModuleNotRunError
from ._typing import array_like, array_check, int_check, List, Optional


def standardization(a: array_like, axis: int = None) -> np.ndarray:
    """
    Standardize for a sequence.

    :param a: array_like
        origin sequence
    :param axis: int
        handle axis
    :return: np.ndarray
        standardization result
    """
    a = array_check(a)
    axis = int_check(axis, nullable=True)

    return (a - np.mean(a, axis=axis, keepdims=True)) / np.std(a, axis=axis, keepdims=True)


def normalization(a: array_like, axis: int = None) -> np.ndarray:
    """
    Normalize for a sequence.

    :param a: array_like
        origin sequence
    :param axis: int
        handle axis
    :return: normalization result
    """
    a = array_check(a)
    axis = int_check(axis, nullable=True)

    return (a - np.min(a, axis=axis)) / (np.max(a, axis=axis) - np.min(a, axis=axis))


def leap_year(year: array_like) -> np.ndarray:
    """
    Calculate a year is leap year or not.

    :param year: array_like
        array like year sequence
    :return: result
    """
    year = np.asarray(year)
    is_leap_year = np.vectorize(_is_leap_year)
    year = is_leap_year(year)

    return np.squeeze(year)


def _is_leap_year(year: int) -> bool:
    """
    # NOTICE: INTERNAL FUNCTION, DO NOT CALL OUTSIDE.
    Calculate a year is leap year or not.

    :param year: input year
    :return: is leap
    """
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


def prepare_empty_container_with_same_size(shape: tuple, number: int) -> List[np.array]:
    """
    :param shape: empty container shape
    :param number: empty container number
    :return:
    """
    return [np.empty(shape=shape) for _ in range(number)]


def hanning_smoothing(a: array_like, border: bool = True) -> np.ndarray:
    """
    Hanning smoothing method
    :param a: array_like
        1-D array
    :param border: bool
        include border value (default True)
    :return: np.ndarray
        smoothing result
    """
    a = array_check(a, 1)

    smoothing, = prepare_empty_container_with_same_size(a.size, 1)

    win = np.array([0.25, 0.5, 0.25])
    for k in range(1, a.size - 1):
        smoothing[k] = np.multiply(win, a[k - 1: k + 2]).sum()

    if border:
        smoothing[0] = a[: 2].mean()
        smoothing[-1] = a[-2:].mean()
    else:
        smoothing = smoothing[1: -1]

    return smoothing


def cov(x: array_like, y: Optional[array_like] = None, rowvar: Optional[bool] = True) -> np.ndarray:
    r"""
    Covariance matrix.

    :param x: array_like
    :param y: optional, array_like
        default is None
    :param rowvar: bool
        If True, x and y is [n_samples, n_features].
        If False, x and y is [n_features, n_samples].
    :math:
        TODO
    :return: np.ndarray
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if not rowvar:
        x = x.T
        y = y.T

    n = x.shape[1]

    if y is not None:
        if x.ndim != y.ndim != 2:
            raise ValueError('Input array must be 2-D.')
        if x.shape != y.shape:
            raise ValueError(f'Input array must have same size, but got {x.shape} and {y.shape}.')

        result = np.dot(x, y.T) / n
    else:
        if x.ndim != 2:
            raise ValueError('Input array must be 2-D.')

        result = np.dot(x, x.T) / n

    return result


# import helper goes following
def _import_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise ImportError('Matplotlib is not found.')

    return plt


class StatisticsModule:
    def __init__(self):
        self._is_fit = False

    def __getattribute__(self, item):
        if item == 'fit':
            self._is_fit = True
        return object.__getattribute__(self, item)
