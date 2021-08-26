import numpy as np
from dataclasses import dataclass
from ._typing import array_cross_check, array_like
from .utils import prepare_empty_container_with_same_size


def mean_error(f: array_like, o: array_like) -> float:
    """
    Mean error, input two array must have the same size.

    :param f: 1-D array
        predict array
    :param o: 1-D array
        ture array
    :math:
        TODO
    :return: float
        error value
    """
    f, o = array_cross_check(f, o, 1)

    return (f - o).sum() / f.size


def mean_absolute_error(f: array_like, o: array_like) -> float:
    """
    Mean absolute error, input two array must have the same size.

    :param f: 1-D array
        predict array
    :param o: 1-D array
        ture array
    :math:
        TODO
    :return: float
        error value
    """
    f, o = array_cross_check(f, o, 1)

    return np.abs(f - o).sum() / f.size


def relative_absolute_error(f: array_like, o: array_like) -> float:
    """
    Relative absolute error, input two array must have the same size.

    :param f: 1-D array
        predict array
    :param o: 1-D array
        ture array
    :math:
        TODO
    :return: float
        error value
    """
    f, o = array_cross_check(f, o, 1)

    return np.abs((f - o) / o).sum() / f.size


def root_mean_squared_error(f: array_like, o: array_like) -> float:
    """
    Root mean squared error, input two array must have the same size.

    :param f: 1-D array
        predict array
    :param o: 1-D array
        ture array
    :math:
        TODO
    :return: float
        error value
    """
    f, o = array_cross_check(f, o)

    return np.sqrt(np.nansum(np.power(f - o, 2)) / np.logical_not(np.isnan(f - o)).sum())


@dataclass
class PrecipitationIndex:
    """
    Precipitation index container
    """
    TS: np.array
    PO: np.array
    FAR: np.array
    B: np.array

    R: np.array
    ETS: np.array


def precipitation(f: array_like, o: array_like) -> PrecipitationIndex:
    """
    Calculate precipitation index.

    :param o: 1-D array
        observation
    :param f: 1-D array
        forecast value
    :math:
        TODO
    :return: PrecipitionIndex
         index included TS, PO, TAR, B, R and ETS
    """
    o, f = array_cross_check(o, f)

    area = np.array([0, 0.1, 10.0, 25.0, 50.0, 100.0, np.inf])
    ts, po, far, b, r, ets = prepare_empty_container_with_same_size(area.shape, 6)
    for index, a in enumerate(area[1:-1], 1):
        ka, kb, kc, kd = 0, 0, 0, 0

        for i in range(o.size):

            if (area[index - 1] <= o[i] < area[index]) \
                    and (area[index - 1] <= f[i] < area[index]):
                ka += 1
            elif not (area[index - 1] <= o[i] < area[index]) \
                    and (area[index - 1] <= f[i] < area[index]):
                kb += 1
            elif (area[index - 1] <= o[i] < area[index]) \
                    and not (area[index - 1] <= f[i] < area[index]):
                kc += 1
            elif o[i] == 0. and f[i] == 0:
                kd += 1

        # TS
        ts[index] = 0 if ka + kb + kc == 0 else ka / (ka + kb + kc)
        # PO
        po[index] = 0 if ka + kc == 0 else kc / (ka + kc)
        # FAR
        far[index] = 0 if ka + kb == 0 else kb / (ka + kb)
        # B
        b[index] = 0 if ka + kc == 0 else (ka + kb) / (ka + kc)
        # R
        r[index] = 0 if ka + kc + kc + kd == 0 else (ka + kb) * (ka + kc) / (ka + kc + kc + kd)
        # ETS
        ets[index] = 0 if ka + kc + kc + r[index] == 0 else (ka - r[index]) / (ka + kc + kc + r[index])

    return PrecipitationIndex(TS=ts[1: -1], PO=po[1: -1], FAR=far[1: -1], B=b[1: -1], R=r[1: -1], ETS=ets[1: -1])
