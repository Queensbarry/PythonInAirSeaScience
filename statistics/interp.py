import numpy as np
from scipy.interpolate import interp1d, interp2d
from typing import Callable
from geographiclib.geodesic import Geodesic

from ._typing import array_check, array_cross_check, array_like, Number, NumberOrArray, Tuple, Dict, Callable
from .utils import prepare_empty_container_with_same_size
from ..utils.geography import distance


class Hermite:
    """
    Hermite interpolate method.
    """
    def __init__(self, x: array_like, y: array_like, dy: array_like) -> None:
        """
        :param x: array_like
            `x` represents the x-coordinates of a set of data points.
        :param y: array_like
            `y` represents the y-coordinates of a set of data points.
        :param dy:
            `dy` represents the derivative of a set of data points.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        dy = np.asarray(dy)

        if x.ndim != y.ndim != dy.ndim != 1:
            raise ValueError('Input array must be 1-D.')

        if x.size != y.size != dy.size:
            raise ValueError('Input array must have same size.')

        self.x = x
        self.y = y
        self.dy = dy

    def __call__(self, x_new: array_like) -> NumberOrArray:
        """
        Calculate new `y` value by `x``.

        :param x_new: array_like
            New `x` value for calculate.
        :return: {int, float, np.ndarray}
            Interpolate value.
        """
        interp_f = np.vectorize(self._interp, otypes=[np.float])

        x_new = np.asarray(x_new)
        y_new = interp_f(x_new)

        return np.squeeze(y_new)

    def _interp(self, x_new: Number) -> NumberOrArray:
        r"""
        NOTICE: Internal function, do not call outside.

        Calculate new `x` value in new sequence.

        :param x_new: Number
            `x` value in new sequence.
        :math:
            TODO
        :return: Number
            `y` value in new sequence.
        """

        dl_f = np.vectorize(self._dl)
        dl_v = dl_f(self.x)

        l_f = np.vectorize(self._l, excluded=['x_new'], otypes=[np.float])
        l_v = l_f(self.x, x_new)

        return np.sum(
            (self.y + (x_new - self.x) * (self.dy - 2 * self.y * dl_v)) *
            np.power(l_v, 2)
        )

    def _l(self, x: Number, x_new: Number) -> NumberOrArray:
        r"""
        NOTICE: Internal function, do not call outside.

        Calculate `l`.

        :param x: Number
            `x` value in period array.
        :param x_new: Number
            `x` value for l calculation, it is a new value which is provided by user.
        :math:
            TODO
        :return: NumberOrArray
            l result, it is a number.
        """
        divide = (x_new - self.x) / (x - self.x)
        divide[np.logical_or(divide == np.inf, divide == -np.inf)] = np.nan

        return np.nanprod(divide)

    def _dl(self, x: Number) -> Number:
        r"""
        NOTICE: Internal function, do not call outside.

        Calculate `dl`.

        :param x: {int, float}
            `x` value for derivative calculation.
        :math:
            TODO
        :return: dl result
        """
        reciprocal = np.reciprocal(x - self.x)
        reciprocal[np.logical_or(reciprocal == np.inf, reciprocal == -np.inf)] = np.nan

        return np.nansum(reciprocal)


class Aitken:
    """
    Aitken interpolation method.
    """
    def __init__(self, x: array_like, y: array_like, eps: Number = 1e-6):
        """
        :param x: array_like
            1-D array
        :param y: array_like
            1-D array
        """
        x, y = array_cross_check(x, y, 1)

        # sort
        self.x = x[np.argsort(x)]
        self.y = y[np.argsort(x)]

        # estimated value
        self._f = np.empty(shape=x.shape)

        self.eps = eps

        self._m = None

    def __call__(self, x_new: array_like):

        x_new = array_check(x_new, (0, 1))

        # calculate point number
        n = self.x.size
        m = 10 if n > 10 else n

        # single number array
        if n == 1:
            return self.y

        self._m = m

        aitken = np.vectorize(self._aitken)

        return np.squeeze(aitken(x_new))

    def _aitken(self, x_new: Number):

        nearest = np.argsort(np.abs(self.x - x_new))
        xm = self.x[nearest][: self._m]
        ym = self.y[nearest][: self._m]

        z = 0
        for i in np.arange(1, self._m):
            z = ym[i]
            for j in np.arange(1, i + 1):
                z = ym[j - 1] + (x_new - xm[j - 1]) * (ym[j - 1] - z) / (xm[j - 1] - xm[i])
            ym[i] = z

            eps = np.abs(ym[i] - ym[i - 1])

            if eps < self.eps:
                break

        return z


class CubicSplineFunction:
    """
    Cubic spline function with different boundary condition.
    Totally, there are 3 boundary condition in this function.

    1st boundary condition:
        provide `x`, `y` and `first-order derivative`.
    2nd boundary condition:
        provide `x`, `y` and `second-order derivative`.
    3th boundary condition:
        provide `x`, `y`.
    """

    ALLOW_CONDITION = [1, 2, 3]
    CONDITION_WITH_D = [1, 2]

    _prepare_empty_container_with_same_size = staticmethod(prepare_empty_container_with_same_size)

    def __init__(self, x: array_like, y: array_like, *, condition: int, **kwargs) -> None:
        """
        :param x: array_like
            `x` represents the x-coordinates of a set of data points.
        :param y: array_like
            `y` represents the x-coordinates of a set of data points.
        :param condition: int
            `condition` represents the boundary condition.
            Mapping:
                1 -- 1st boundary condition
                2 -- 2nd boundary condition
                3 -- 3th boundary condition
        :param kwargs:
            `d` represents derivative value.
                If condition is 1, provide first-order derivative.
                If condition is 2, provide second-order derivative.
        """
        if condition not in self.ALLOW_CONDITION:
            raise ValueError('Condition is not allow.')

        if (condition in self.CONDITION_WITH_D) and (kwargs.get('d') is None):
            raise ValueError(f'In condition `{condition}`, you need to provide `d` in initial method.')

        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim != y.ndim:
            raise ValueError('Input array must be 1-D array.')

        if x.size != y.size:
            raise ValueError('Input array must have same size.')

        d = None
        if condition in self.CONDITION_WITH_D:
            d = np.asarray(kwargs.get('d'))
            if (d.ndim != 1) and (d.size != 2):
                raise ValueError('Input params `d` is invalid.')

        self.x = x
        self.y = y
        self.d = d

        self.condition = condition

    def __call__(self, x_new: array_like) -> NumberOrArray:
        """
        Calculate new `y` value by new `x` value.

        :param x_new: array_like
            `x` value for interpolation calculation.
        :return: NumberOrArray
            Interpolation result.
        """
        x_new = np.asarray(x_new)
        mapping: Dict[int, Callable] = {
            1: np.vectorize(self._first),
            2: np.vectorize(self._second),
            3: np.vectorize(self._third)
        }
        f = mapping[self.condition]
        values = f(x_new)

        return np.squeeze(values)

    def _first(self, x_new: Number):
        a, b, h, alpha, beta, dy = self._prepare_empty_container_with_same_size(self.x.shape, 6)

        a[0], a[-1] = 0, np.nan
        b[0], b[-1] = self.d[0], np.nan
        h[-1] = np.nan
        alpha[0], alpha[-1] = np.nan, np.nan
        beta[0], beta[-1] = np.nan, np.nan
        dy[0], dy[-1] = self.d[0], self.d[1]

        h, alpha, beta, a = self._calculate_h_alpha_beta_a(self.x, self.y, h, alpha, beta, a)
        b = self._calculate_bj(b, beta, alpha, a)

        for j in np.arange(self.x.size - 2, 0, -1):
            dy[j] = a[j] * dy[j + 1] + b[j]

        s_i, s_p1_i, s, s_p1 = self._find_nearest_index_and_value(self.x, x_new)

        return self._calculate_y_new(x_new, self.y, h, dy, s_i, s_p1_i, s, s_p1)

    def _second(self, x_new: Number):
        a, b, h, alpha, beta, dy = self._prepare_empty_container_with_same_size(self.x.shape, 6)

        a[0], a[-1] = -0.5, np.nan
        b[0] = 1.5 * (self.y[1] - self.y[0]) / (self.x[1] - self.x[0]) - (self.x[1] - self.x[0]) * self.d[0] / 4
        b[-1] = np.nan

        h, alpha, beta, a = self._calculate_h_alpha_beta_a(self.x, self.y, h, alpha, beta, a)
        b = self._calculate_bj(b, beta, alpha, a)
        dy[-1] = (3 * (self.y[-1] - self.y[-2]) / h[-2] + self.d[1] * h[-2] / 2 - b[-2]) / (2 + a[-2])

        for j in np.arange(self.x.size - 2, -1, -1):
            dy[j] = a[j] * dy[j + 1] + b[j]
        s_i, s_p1_i, s, s_p1 = self._find_nearest_index_and_value(self.x, x_new)

        return self._calculate_y_new(x_new, self.y, h, dy, s_i, s_p1_i, s, s_p1)

    def _third(self, x_new: Number):
        a, b, c, h, alpha, beta, dy, t, v = self._prepare_empty_container_with_same_size(self.x.shape, 9)

        a[0], a[-1] = 0, np.nan
        b[0], b[-1] = 1, np.nan
        c[0], c[-1] = 0, np.nan
        t[-1] = 0
        v[-1] = 0

        h, alpha, beta, a = self._calculate_h_alpha_beta_a(self.x, self.y, h, alpha, beta, a)
        # b_j
        for j, item in enumerate(np.nditer(b[1: -1], op_flags=['readwrite']), 1):
            item[...] = -1 * (1 - alpha[j - 1]) * b[j - 1] / (2 + (1 - a[j - 1]) * a[j - 1])
        # c_j
        for j, item in enumerate(np.nditer(c[1: -1], op_flags=['readwrite']), 1):
            item[...] = (beta[j - 1] - (1 - alpha[j - 1]) * c[j - 1]) / (2 + (1 - a[j - 1]) * a[j - 1])

        # t_j
        for j in np.arange(self.x.size - 2, 0, -1):
            t[j] = a[j] * t[j + 1] + b[j]
        # v_j
        for j in np.arange(self.x.size - 2, 0, -1):
            v[j] = a[j] * v[j + 1] + c[j]

        # dy_{n - 1}
        dy[-2] = (beta[-2] - alpha[-2] * v[1] - (1 - alpha[-2]) * v[-2]) / \
                 (2 + alpha[-2] * t[1] + (1 - alpha[-2]) * t[-2])
        # dy_j
        for j in range(0, self.x.size - 2):
            dy[j] = t[j + 1] * dy[-2] + v[j + 1]
        dy[-1] = dy[0]

        s_i, s_p1_i, s, s_p1 = self._find_nearest_index_and_value(self.x, x_new)

        return self._calculate_y_new(x_new, self.y, h, dy, s_i, s_p1_i, s, s_p1)

    @staticmethod
    def _calculate_h_alpha_beta_a(
            x: np.ndarray, y: np.ndarray,
            h: np.ndarray, alpha: np.ndarray, beta: np.ndarray, a: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        NOTICE: Internal function, do not call outside.

        Calculate params `h`, `alpha`, `beta`, `a` for every boundary condition.

        :param x: np.ndarray
            `x` value sequence
        :param y: np.ndarray
            `y` value sequence
        :param h: np.ndarray
            empty container
            :math:
                TODO
        :param alpha: np.ndarray
            empty container
            :math:
                TODO
        :param beta: np.ndarray
            empty container
            :math:
                TODO
        :param a: np.ndarray
            empty container
            :math:
                TODO
        :return: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            result value order by `h`, `alpha`, `beta`, `a`
        """
        h[: -1] = x[1:] - x[: -1]
        alpha[1: -1] = h[: -2] / (h[: -2] + h[1: -1])
        beta[1: -1] = 3 * (
                (1 - alpha[1: -1]) * (y[1: -1] - y[: -2]) / h[: -2]
                + alpha[1: -1] * (y[2:] - y[1: -1]) / h[1: -1]
        )
        for j, item in enumerate(np.nditer(a[1: -1], op_flags=['readwrite']), 1):
            item[...] = -1 * alpha[j] / (2 + (1 - alpha[j]) * a[j - 1])

        return h, alpha, beta, a

    @staticmethod
    def _calculate_bj(
            b: np.ndarray, beta: np.ndarray, alpha: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        r"""
        NOTICE: Internal function, do not call outside.

        Calculate params `b` for 1st and 2nd boundary condition.

        :param b:
            empty container
            :math:
                TODO
        :param beta:
            array with value
        :param alpha:
            array with value
        :param a:
            array with value
        :return: np.ndarray
            value `b` result
        """
        for j, item in enumerate(np.nditer(b[1: -1], op_flags=['readwrite']), 1):
            item[...] = (beta[j] - (1 - alpha[j]) * b[j - 1]) / (2 + (1 - alpha[j]) * a[j - 1])

        return b

    @staticmethod
    def _calculate_y_new(
            x_new: float, y: np.ndarray,
            h: np.ndarray, dy: np.ndarray,
            s_i: int, s_p1_i: int, s: float, s_p1: float
    ) -> float:
        r"""
        NOTICE: Internal function, do not call outside.

        :param x_new: float
            value `x` new
        :param y: np.ndarray
            value `y` sequence
        :param h: np.ndarray
            value `h` sequence
        :param dy: np.ndarray
             value `dy` sequence
        :param s_i: int
            small index
        :param s_p1_i: int
            big index
        :param s: float
            small index -> value
        :param s_p1:
            big index -> value
        :math:
            TODO
        :return: float
            new `y` value
        """
        y_new = \
            (3 * (s_p1 - x_new) ** 2 / h[s_i] ** 2 - 2 * (s_p1 - x_new) ** 3 / h[s_i] ** 3) * y[s_i] + \
            (3 * (x_new - s) ** 2 / h[s_i] ** 2 - 2 * (x_new - s) ** 3 / h[s_i] ** 3) * y[s_p1_i] + \
            h[s_i] * ((s_p1 - x_new) ** 2 / h[s_i] ** 2 - (s_p1 - x_new) ** 3 / h[s_i] ** 3) * dy[s_i] - \
            h[s_i] * ((x_new - s) ** 2 / h[s_i] ** 2 - (x_new - s) ** 3 / h[s_i] ** 3) * dy[s_p1_i]

        return y_new

    @staticmethod
    def _find_nearest_index_and_value(
            x: np.ndarray, x_new: float
    ) -> Tuple[int, int, float, float]:
        """
        NOTICE: Internal function, do not call outside.

        Find nearest index and value by new x.

        :param x: np.ndarray
            value `x` sequence
        :param x_new: float
            new `x` for find index
        :return: Tuple[int, int, float, float]
            represents [index, index, value, value]
        """
        diff = x - x_new
        nearest_index = np.argsort(np.abs(diff))[: 2]

        nearest = x[nearest_index]
        s_i, s_p1_i = nearest_index[np.argsort(nearest)]
        s, s_p1 = nearest[np.argsort(nearest)]

        return s_i, s_p1_i, s, s_p1


class TripleVarTwoPoint:

    def __init__(self, x: np.array, y: np.array, z: np.array):

        self.x = array_check(x, 1)
        self.y = array_check(y, 1)
        self.z = array_check(z, 2)

        # TODO: array cross check before get the value to variable

    def __call__(self, x_new: float, y_new: float):

        x_new = np.asarray(x_new)
        y_new = np.asarray(y_new)

        cal = np.vectorize(self._cal)

        return np.squeeze(cal(x_new, y_new))

    def _cal(self, x_new: Number, y_new: Number):

        p = np.where(np.abs(self.x - x_new) == np.abs(self.x - x_new).min())[0][0] - 1
        q = np.where(np.abs(self.y - y_new) == np.abs(self.y - y_new).min())[0][0] - 1

        sum_ = 0
        for pi in range(p, p + 3):
            for qi in range(q, q + 3):
                _p = np.empty(shape=(3,))
                _q = np.empty(shape=(3,))
                for i, p_ in enumerate(range(p, p + 3)):
                    if pi != p_:
                        _p[i] = (x_new - self.x[p_]) / (self.x[pi] - self.x[p_])
                    else:
                        _p[i] = np.nan
                for i, q_ in enumerate(range(q, q + 3)):
                    if qi != q_:
                        _q[i] = (y_new - self.y[q_]) / (self.y[qi] - self.y[q_])
                    else:
                        _q[i] = np.nan
                p_prod = np.nanprod(_p)
                q_prod = np.nanprod(_q)
                sum_ += p_prod * q_prod * self.z[pi, qi]

        return sum_


class ReverseDistance:

    def __init__(self,
        lon: np.array, lat: np.array,
        lon1: float, lon2: float,
        lat1: float, lat2: float,
        d_lon: float, d_lat: float,
        r: float, v: np.array
    ):

        # TODO: type check
        self.lon = lon
        self.lat = lat
        self.lon1, self.lon2 = lon1, lon2
        self.lat1, self.lat2 = lat1, lat2
        self.d_lon, self.d_lat = d_lon, d_lat
        self.r = r
        self.v = v

    def reverse_distance(self):

        return self._cal(
            self.lon, self.lat,
            self.lon1, self.lon2, self.lat1, self.lat2,
            self.d_lon, self.d_lat, self.r, self.v
        )

    def modify(self, m: int):
        # TODO: change language to English
        if m < 1:
            raise ValueError

        result = None
        for _ in range(m):
            # 正插
            result = self._cal(
                self.lon, self.lat,
                self.lon1, self.lon2, self.lat1, self.lat2,
                self.d_lon, self.d_lat,
                self.r, self.v
            )
            # 反插
            station_result = self._interp2d4geo(
                np.arange(self.lon1, self.lon2 + self.d_lon, self.d_lon),
                np.arange(self.lat1, self.lat2 + self.d_lat, self.d_lat),
                self.lon, self.lat, result)
            # 测站差值
            diff = self.v - station_result
            # 差值正插
            diff_result = self._cal(
                self.lon, self.lat,
                self.lon1, self.lon2, self.lat1, self.lat2,
                self.d_lon, self.d_lat,
                self.r, diff
            )
            result += diff_result

        return result

    @staticmethod
    def _cal(
            lon: np.array, lat: np.array,
            lon1: float, lon2: float,
            lat1: float, lat2: float,
            d_lon: float, d_lat: float,
            r: float, v: np.array) -> np.array:

        lon_ = np.arange(lon1, lon2 + d_lon, d_lon)
        lat_ = np.arange(lat1, lat2 + d_lat, d_lat)

        x, y = np.meshgrid(lon_, lat_)

        out = np.empty(shape=x.shape)
        for item in np.nditer(
                [x, y, out], [],
                [['readonly'], ['readonly'], ['writeonly', 'allocate']]):
            w = np.zeros(shape=v.shape)
            for station in np.nditer([lon, lat, w], [], [['readonly'], ['readonly'], ['writeonly']]):
                # km
                distance_ = distance((item[1], item[0]), (station[1], station[0]))
                if distance_ <= r:
                    station[2][...] = (distance_ ** 2 - r ** 2) / (distance_ ** 2 + r ** 2)
            item[2][...] = (v * w).sum() / w.sum()

        return out

    @staticmethod
    def _interp2d4geo(lon: np.array, lat: np.array, x: np.array, y: np.array, v: np.array):
        # new_v = np.zeros(shape=x.shape)
        # for item in np.nditer([x, y, new_v], [], [['readonly'], ['readonly'], ['writeonly']]):
        #     x_i = np.where(lon <= item[0])[0].max()
        #     y_i = np.where(lat <= item[1])[0].max()
        #     x_ = np.array([0, distance((lat[y_i], lon[x_i]), (lat[y_i], lon[x_i + 1]))])
        #     f = interp1d(x_, np.array([v[y_i, x_i], v[y_i, x_i + 1]]))
        #     upper = f(distance((lat[y_i], lon[x_i]), (lat[y_i], item[0])))
        #
        #     x_ = np.array([0, distance((lat[y_i - 1], lon[x_i]), (lat[y_i - 1], lon[x_i + 1]))])
        #     f = interp1d(x_, np.array([v[y_i - 1, x_i], v[y_i - 1, x_i + 1]]))
        #     under = f(distance((lat[y_i - 1], lon[x_i]), (lat[y_i - 1], item[0])))
        #
        #     y_ = np.array([0, distance((lat[y_i], lon[x_i]), (lat[y_i - 1], lon[x_i]))])
        #     f = interp1d(y_, np.array([upper, under]))
        #     item[2][...] = f(distance((lat[y_i], lon[x_i]), (item[1], lon[x_i])))
        #
        # return new_v

        f = interp2d(lon, lat, v)

        return f(x, y)


class Newton:
    def __init__(self, x: array_like, y: array_like):
        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError('Input array must be 1-D.')

        if (x.size != y.size) or (x.size < 2):
            raise ValueError(
                f'x, y must have the same shape, and its size must bigger than 2, but got {x.shape}, {y.shape}'
            )

        self.x = x
        self.y = y

    def __call__(self, x: array_like):
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError('Input array must be 1-D.')
        newton = np.vectorize(self._newton)

        return newton(x)

    def _newton(self, x: float):
        """
        Inner function, do not call it outside.
        :param x: interpolation single value
        :return: interpolation result
        """
        result = self.y[0]
        for i in range(2, self.x.size):
            result += self._diff_quot(self.x[: i], self.y[: i]) * self._w(x, i - 1)

        return result

    def _diff_quot(self, x: np.array, y: np.array):
        """
        Inner function, do not call it outside.
        :param x: x array
        :param y: y array
        :return: diff quot value
        """
        if x.size > 2:
            return (self._diff_quot(x[: -1], y[: -1]) - self._diff_quot(x[1:], y[1:])) / (x[0] - x[-1])

        return (y[0] - y[1]) / (x[0] - x[1])

    def _w(self, x: float, i: int):
        """
        Inner function, do not call it outside.
        :param x: interpolation value
        :param i: order
        :return:
            Example:
                w1 = (x - x0)
                w2 = (x - x0)(x - x1)
                w3 = (x - x0)(x - x1)(x - x2)
        """
        return np.prod(x - self.x[: i])
