import numpy as np
from ._typing import array_cross_check, array_like, number_check, Number
from .utils import prepare_empty_container_with_same_size
from ..utils import geography


class Turbulence:

    distance = staticmethod(geography.distance)

    def __init__(self,
                 u: array_like, v: array_like,
                 lon: array_like, lat: array_like,
                 dz: Number = 300):

        u, v = array_cross_check(u, v)
        lon, lat = array_cross_check(lon, lat)

        u, lon = array_cross_check(u, lon)

        self.u = u
        self.v = v

        self.s = np.sqrt(u ** 2 + v ** 2)

        self.lon = lon
        self.lat = lat

        self.dz = dz

        self._du = None
        self._dv = None

        self._dx = None
        self._dy = None

        self._ds = None

    def _du_dv(self):
        if (self._du is None) or (self._dv is None):
            du = np.diff(self.u, axis=1)[:-1, :]
            dv = np.diff(self.v, axis=0)[:, :-1]

            self._du, self._dv = du, dv

        return self._du, self._dv

    def _dx_dy(self):
        if (self._dx is None) or (self._dy is None):
            m, n = self.lon.shape
            dx, dy = prepare_empty_container_with_same_size((m - 1, n - 1), 2)
            for x in range(1, m):
                for y in range(1, n):
                    _dx = self.distance((y, x), (y, x + 1), unit=geography.Unit.METERS)
                    _dy = self.distance((y, x), (y + 1, x), unit=geography.Unit.METERS)

                    dx[x - 1, y - 1] = _dx
                    dy[x - 1, y - 1] = _dy

            self._dx, self._dy = dx, dy

        return self._dx, self._dy

    def _cal_ds(self):
        if self._ds is None:
            self._ds = np.diff(self.u, axis=1)[:-1, :]

        return self._ds

    @property
    def vws(self):
        du, dv = self._du_dv()

        return np.sqrt(np.abs(du / self.dz) ** 2 + np.abs(dv / self.dz) ** 2)

    @property
    def def_(self):
        du, dv = self._du_dv()
        dx, dy = self._dx_dy()

        return np.sqrt((dv / dx + du / dy) ** 2 + (du / dx - dv / dy) ** 2)

    @property
    def div(self):
        du, dv = self._du_dv()
        dx, dy = self._dx_dy()

        return np.sqrt(du / dx - dv / dy)

    @property
    def ei(self):

        return self.vws * (self.def_ + self.div)

    @property
    def ti(self):

        return self.vws * (self.def_ - self.div)

    @property
    def mos(self):

        return np.abs(self.u[1:, 1:]) * self.def_

    @property
    def hws(self):
        dx, dy = self._dx_dy()
        ds = self._cal_ds()

        u = self.u[1:, 1:]
        v = self.v[1:, 1:]
        s = self.s[1:, 1:]

        return (u / s) * (ds / dy) + (v / s) * (ds / dx)

    @property
    def button(self):

        return 1.25 * self.hws + 0.25 * self.vws + 10.5


class Icing:

    @staticmethod
    def icao(rh: Number, temperature: Number):
        rh = number_check(rh, 0, 100)
        t = number_check(temperature)

        icing = None
        ic = 2 * (rh - 50) * (t * (t + 14) / (-49)) / 10

        if 0 <= ic < 4:
            icing = 1
        elif 4 <= ic < 7:
            icing = 2
        elif 7 <= ic:
            icing = 3

        return ic, icing

    @classmethod
    def new(cls, rh: Number, temperature: Number, omega: float):
        omega = number_check(omega)

        ic = cls.icao(rh, temperature)
        icing = ic[1] if omega <= -0.2 else None

        return ic[0], icing

    @staticmethod
    def raob(temperature: Number, dew_point: Number, gamma: Number):

        t = number_check(temperature)
        dew_point = number_check(dew_point)
        gamma = number_check(gamma)

        diff = t - dew_point

        icing = None
        if -8 < t <= 0:
            if diff <= 1:
                icing = 3 if gamma <= 2 else 7
            elif 1 < diff < 3:
                icing = 1 if gamma <= 2 else 4
        elif -16 < t <= -8:
            if diff <= 1:
                icing = 6 if gamma <= 2 else 5
            elif 1 < diff < 3:
                icing = 3 if gamma <= 2 else 2
        elif -22 < t <= -16:
            icing = 3

        return icing

    @staticmethod
    def assume_frost_point(temperature: Number, dew_point: Number, v: Number):

        t = number_check(temperature)
        td = number_check(dew_point)
        v = number_check(v)

        _index = -0.05 * (v / 100)**2

        icing = None
        tfi = _index * (t - td)
        diff = tfi - t
        if diff > _index:
            icing = 2 if diff > 0 else 1

        return icing
