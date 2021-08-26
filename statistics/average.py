import numpy as np
import pandas as pd
from ._typing import array_like, array_check, int_check, Optional
from ..utils.geography import area


def depature(a: array_like, axis: Optional[int] = None) -> np.ndarray:
    """
    Calculate a depature value with a sequence.

    :param a: array_like
        sequence
    :param axis: optional, int
        handle axis
    :return: np.ndarray
        depature result
    """
    a = array_check(a)
    axis = int_check(axis, nullable=True)
    mean = a.mean(axis=axis, keepdims=True)

    return a - mean


class ClimateState:
    """
    Calculate climate state.
    """
    def __init__(self, *,
                 unit: Optional[str] = 'h', origin: Optional[str] = '1800-01-01',
                 start: Optional[str] = None, end: Optional[str] = None) -> None:
        """
        :param unit: optional, str
            time sequence unit, default is hour.
        :param origin: optional, str
            time sequence origin
        :param start: optional, str
            time range start, default None mean that from the data start time
        :param end: optional, str
            time range end, default None mean that to the data end time
        """
        self.unit = unit
        self.origin = origin
        self.start = start
        self.end = end

        self._climate_state = None
        self._depature = None

    def __call__(self, a: array_like, t: array_like, axis: Optional[int] = 0):
        """
        :param a: array_like
            3-D array with data
        :param t: array_like
            1-D array with time
        :param axis: optional, int
            handle axis
        :return: class
            self
        """
        a = array_check(a, 3)
        t = array_check(t, 1)
        axis = int_check(axis)

        df = pd.DataFrame(np.fromfunction(lambda x: x, shape=t.shape, dtype=np.int), index=t)
        df.index = pd.to_datetime(df.index, unit=self.unit, origin=pd.Timestamp(self.origin))
        if self.start is None:
            self.start = df.iloc[[0]].index[0]
        if self.end is None:
            self.end = df.iloc[[-1]].index[0]

        # split df from start time to end time
        df = df[self.start: self.end]
        group_result = df.groupby(lambda x: x.month)

        result = np.full((12, *a.shape[1:]), np.nan)
        for index, d in group_result:
            result[index - 1] = a[d[0].values].mean(axis=axis)
        self._climate_state = result

        # sort value in time range for avoiding index error
        depature_ = np.empty(shape=a[df[0].to_numpy()].shape)
        for i, _ in enumerate(depature_):
            depature_[i] = a[i] - df.iloc[[i]].index.month[0]
        self._depature = depature_

        return self

    @property
    def climate_state(self) -> np.ndarray:
        """
        :return: np.ndarray
            climate state
        """
        if self._climate_state is None:
            raise ValueError('Please execute the model first')

        return self._climate_state

    @property
    def depature(self) -> np.ndarray:
        """
        :return: np.ndarray
            depature value
        """
        if self._depature is None:
            raise ValueError('Please execute the model first')

        return self._depature


class Nino:
    """
    Calculate nino index.
    """
    def __init__(self, *,
                 unit: Optional[str] = 'D', origin: Optional[str] = '1800-01-01 00:00:00',
                 start: Optional[str] = None, end: Optional[str] = None) -> None:
        """
        :param unit: str
            time sequence unit, default is hour.
        :param origin: str
            time sequence origin
        :param start: str
            time range start
        :param end: str
            time range end
        """
        self.unit = unit
        self.origin = origin
        self.start = start
        self.end = end

        self.cs = ClimateState(unit=unit, origin=origin, start=start, end=end)

        self._depature = None
        self._nino = None

    def __call__(self,
                 a: array_like, t: array_like,
                 lat: array_like, lon: array_like,
                 axis: Optional[int] = 0,
                 dx: Optional[float] = None, dy: Optional[float] = None):
        """
        :param a: array_like
            3-D array with data
        :param t: array_like
            1-D array with time
        :param lat: array_like
            1-D array
        :param lon: array_like
            1-D array
        :param axis: int
            handle axis
        :param dx: float
            longitude differ, default auto calculate
        :param dy: float
            latitude differ, default auto calculate
        :return: class
            self
        """
        a = array_check(a, 3)
        lon = array_check(lon, 1)
        lat = array_check(lat, 1)
        # execute ClimateState for get depature
        self.cs(a, t, axis)
        acreage = self._nino_area(lat, lon, dx, dy)
        self._depature = self.cs.depature
        self._nino = np.multiply(self._depature, acreage).sum(axis=(1, 2)) / acreage.sum()

        return self

    @property
    def depature(self) -> np.ndarray:
        """
        :return: np.ndarray
            depature value
        """
        if self._depature is None:
            raise ValueError('Please execute the model first')

        return self._depature

    @property
    def nino(self) -> np.ndarray:
        """
        :return:
        """
        if self._nino is None:
            raise ValueError('Please execute the model first')

        return self._nino

    @classmethod
    def _nino_area(cls,
                   lat: np.ndarray, lon: np.ndarray,
                   dx: float = None, dy: float = None) -> np.ndarray:
        """
        NOTICE: INTERNAL FUNCTION, DO NOT CALL OUTSIDE.
        Calculate area by a point

        :param lat: np.ndarray
            1-D array
        :param lon: np.ndarray
            1-D array
        :param dx: float
            longitude differ, default auto calculate
        :param dy: float
            latitude differ, default auto calculate
        :return: np.ndarray
            nino area
        """
        if dx is None:
            dx = np.abs(lat[1] - lat[0])
        if dy is None:
            dy = np.abs(lon[1] - lon[0])
        _lon, _lat = np.meshgrid(lon, lat)

        cal_area = np.frompyfunc(cls._nino_point2area, 4, 1)

        return cal_area(_lon, _lat, dx, dy).astype(np.float)

    @staticmethod
    def _nino_point2area(lon: float, lat: float, dx: float, dy: float) -> float:
        """
        NOTICE: INTERNAL FUNCTION, DO NOT CALL OUTSIDE.

        :param lat: float
            latitude
        :param lon: float
            longitude
        :param dx: float
            longitude differ
        :param dy: float
            latitude differ
        :return: float
            area
        """
        top = lat + dy if lat + dy <= 90 else 90
        bottom = lat - dy if lat - dy >= -90 else -90

        return area(
            [lon - dx, bottom],
            [lon - dx, top],
            [lon + dx, top],
            [lon + dx, bottom],
            [lon - dx, bottom]
        )
