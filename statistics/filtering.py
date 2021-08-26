import math
import numpy as np
from ._typing import array_check, array_like, int_check, number_check, Number
from .tendency import moving_average as ma


def moving_average(a: array_like, win: int) -> np.ndarray:
    """
    Moving average filtering, it is a low pass filter

    :param a: array_like
        original sequence
    :param win: int
        slide length
    :math:
        TODO
    :return: np.ndarray
        filtering result, it length equals (a.size - win + 1)
    """
    return ma(a, win)


def binomial_coefficient(a: array_like, win: int) -> np.ndarray:
    """
    Binomial Coefficient Slide filtering, it is a low pass filter.

    :param a: array_like
        origin sequence
    :param win: int
        sliding windows, it must be `odd`
    :math:
        TODO
    :return: np.ndarray
    """
    a = array_check(a, 1)
    win = int_check(win, order='odd')

    n = win - 1
    factorial = np.vectorize(math.factorial)
    b = np.fromfunction(
        lambda x: factorial(n) / factorial(x) / factorial(n - x),
        shape=(win,)).astype(np.float)

    c = b / b.sum()

    extend_x = np.empty(shape=(a.size + n))
    half = int(n / 2)
    extend_x[: half] = np.flipud(a[: half])
    extend_x[half: -half] = a[:]
    extend_x[-half:] = np.flipud(a[-half:])

    out = np.empty(a.shape)
    for index, item in enumerate(np.nditer(out, op_flags=['writeonly']), half):
        item[...] = np.multiply(extend_x[int(index - n / 2): int(index + n / 2 + 1)], c).sum()

    return out


def recursive_unipolar(a: array_like, win: Number, first: Number = None) -> np.ndarray:
    """
    Recursive unipolar filter is a low pass filter.

    :param a: array
        origin sequence
    :param win: Number
        cycle
    :param first: Number
        In result sequence will not calculate the first value,
        first value will depend on this param.
    :math:
        TODO
    :return: np.ndarray
        filtering result
    """
    a = array_check(a, 1)
    win = number_check(win)
    first = a[0] if first is None else first
    first = number_check(first)

    y = np.empty(a.shape, dtype=np.float)

    # default value
    y[0] = first

    # recursive
    f = 1 / win
    alpha = np.exp(-2 * np.pi * f)
    for index, item in enumerate(np.nditer([a[1:], y[1:]], [], [['readonly'], ['writeonly']]), 1):
        item[1][...] = alpha * item[0] + (1 - alpha) * y[index - 1]

    return y


def gauss(a: array_like, win: int) -> np.ndarray:
    """
    Gauss filter is a low pass filter.

    :param a: array_like
        origin array
    :param win: int
        filtering windows
    :math:
        TODO
    :return: np.ndarray
        filtering result
    """
    a = array_check(a, 1)
    win = int_check(win, 1, order='odd')

    m = int((win - 1) / 2)

    exponent = -(9 * np.power(np.arange(-m, m + 1), 2)) / (2 * m**2)
    c = (3 / (m * np.sqrt(2 * np.pi))) * np.exp(exponent)

    result = np.empty(shape=(a.size - 2 * m,))
    for t in np.arange(m, a.size - m):
        result[t - m] = np.multiply(np.flipud(a[t - m: t + m + 1]), c).sum()

    return result


def lanczos(a: array_like, t1: Number, t2: Number, win: int) -> np.ndarray:
    """
    Lanczos filter is a band pass filter.

    :param a: array_like
        origin array
    :param t1: Number
        small cycle
    :param t2: Number
        bigger cycle
    :param win: int
        filtering windows
    :math:
        TODO
    :return: np.ndarray
        filtering result
    """
    a = array_check(a, 1)
    t1 = number_check(t1)
    t2 = number_check(t2)
    win = int_check(win, 1)

    f1 = 1 / t1
    f2 = 1 / t2

    k = np.arange(-win, win + 1)
    c = (np.sin(2 * np.pi * k * f1) / (np.pi * k) - np.sin(2 * np.pi * k * f2) / (np.pi * k)) * \
        (np.sin(np.pi * k / win) / (np.pi * k / win))
    c[win] = 0

    result = np.empty(shape=(a.size - 2 * win,))
    for t in np.arange(win, a.size - win):
        result[t - win] = np.multiply(np.flipud(a[t - win: t + win + 1]), c).sum()

    return result


def self_designed(a: array_like, freq: Number, win: int) -> np.ndarray:
    """
    Self designed filter is a band pass filter.

    :param a: array_like
        origin sequence
    :param freq: Number
        center frequency
    :param win: Number
        filtering windows
    :math:
        TODO
    :return: np.ndarray
        filtering result
    """
    a = array_check(a, 1)
    freq = number_check(freq, 0)
    win = int_check(win, 0)

    h = np.vectorize(_self_designed_h, otypes=[np.float], excluded=['freq'])

    n = a.size

    f = np.arange(1 / (2 * n), 1 / 2, 1 / (2 * n))
    hf = h(f, freq)
    hk = np.empty(shape=(2 * win + 1,))
    for k in np.arange(-win, win + 1):
        hk[k + win] = (hf * np.cos(2 * np.pi * f * 0)).sum() / n

    result = np.empty(shape=(n - 2 * win,))
    for i in np.arange(win, n - win):
        result[i - win] = np.multiply(np.flipud(a[i - win: i + win + 1]), hk).sum()

    return result


def _self_designed_h(f: Number, freq: Number) -> Number:
    """
    Internal function, do not call outside.

    :param f: Number
        frequency need to calculate
    :param freq: Number
        center frequency
    :math:
        TODO
    :return: Number
        calculation result
    """
    if (0 < f < freq / 2) or (2 * freq <= f < 1 / 2):
        return 0

    result = 0.5
    if freq / 2 <= f < freq:
        result += 0.5 * np.cos(2 * np.pi * f / freq)
    elif freq <= f < 2 * freq:
        result -= 0.5 * np.cos(2 * np.pi * f / (2 * freq))
    else:
        raise ValueError

    return result
