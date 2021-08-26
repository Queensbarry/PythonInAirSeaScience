import numpy as np
import scipy as sp
import scipy.stats
from ._typing import array_check, array_like, int_check, number_check
from .average import depature


def moving_average(a: array_like, step: int, mode: str = 'valid') -> np.ndarray:
    """
    Moving average calculation.

    :param a: array_like
        origin array
    :param step: int
        slide step
    :param mode:
        reference numpy.convolve
    :return: np.ndarray
        return array length is (a.size - step + 1)
    """
    a = array_check(a, 1)
    step = int_check(step, 0)
    moving = np.convolve(a, np.ones((step,)), mode) / step

    return moving


def cum_depature(a: array_like) -> np.ndarray:
    """
    Calculate a cum depature of a 1-D array.

    :param a: array_like
        only 1-D array are allow
    :return: np.ndarray
        cum depature result
    """
    a = array_check(a, 1)

    return depature(a).cumsum()


def smooth(a: array_like, condition: int) -> np.ndarray:
    """
    Smooth array by 5, 7, 9 points.

    :param a: array_like
        only 1-D array are available.
    :param condition: int
        chosen of (5, 7, 9)
    :return: np.ndarray
    """
    a = array_check(a, 1)
    condition = int_check(condition, chosen=(5, 7, 9))

    smooth_option = {
        5: (np.array((-3, 12, 17, 12, -3)), 35),
        7: (np.array((-2, 3, 6, 7, 6, 3, -2)), 21),
        9: (np.array((-21, 14, 39, 54, 59, 54, 39, 14, -21)), 231)
    }

    result = np.empty(shape=a.shape)

    half = (condition - 1) / 2

    for i, in np.ndindex(result.shape):
        if 0 <= i < half:
            result[i] = a[: i + 1].mean()
        elif half <= i <= a.size - half - 1:
            result[i] = np.multiply(
                a[int(i - half): int(i + half + 1)],
                smooth_option[condition][0]
            ).sum() / smooth_option[condition][1]
        elif i > a.size - half - 1:
            result[i] = a[i:].mean()

    return result


def five_point_cubic_smooth(a: array_like) -> np.ndarray:
    """
    Five point cubic smooth

    :param a: array_like
        1-D array
    :return: np.ndarray
    """
    a = array_check(a, 1)
    new_a = np.empty(shape=a.shape)

    new_a[0] = np.multiply(a[: 5], np.array([69, 4, -6, 4, -1])).sum() / 70
    new_a[1] = np.multiply(a[: 5], np.array([2, 27, 12, -8, -1])).sum() / 35

    new_a[-1] = np.multiply(a[-5:], np.array([-1, 4, -6, 4, 69])).sum() / 70
    new_a[-2] = np.multiply(a[-5:], np.array([-1, -8, 12, 27, 2])).sum() / 35
    for index, item in enumerate(np.nditer(new_a[2: -2], op_flags=['writeonly']), 2):
        item[...] = np.multiply(a[index - 2: index + 3], np.array([-3, 12, 17, 12, -3])).sum() / 35

    return new_a


def significance_test(a: array_like, alpha: float) -> bool:
    a = array_check(a, 1)
    alpha = number_check(alpha, 0, 1)
    tau = sp.stats.kendalltau(np.arange(a.size), a)[0]
    n = a.size
    var_tau = (4 * n + 10) / 9 / n / (n - 1)
    u = tau / np.power(var_tau, 0.5)

    return np.abs(u) > sp.stats.norm.isf(alpha / 2)
