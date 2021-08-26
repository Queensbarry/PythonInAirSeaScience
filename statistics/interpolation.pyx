import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc
from .utils import prepare_empty_container_with_same_size

# cdef tuple prepare_empty_container_with_same_size(tuple shape, int number):
#     pass


cdef class CubicInterpolation:

    cdef:
        np.ndarray x, y
        int type_
    cdef Py_ssize_t n

    def __init__(self, np.ndarray[np.double_t, ndim=1] x, np.ndarray[np.double_t, ndim=1] y, int type_):

        cdef list _t = [1, 2, 3]
        if type_ not in _t:
            raise ValueError('Type must be one of 1, 2 or 3.')

        self.x = x
        self.y = y
        self.type_ = type_

        self.n = x.shape[0]

    def __call__(self, x, **kwargs):

        cdef np.ndarray[double, ndim=1] x_new = np.asarray(x, dtype=np.float)

        cdef double d[2]

        if self.type_ == 1:
            if kwargs.get('d') is None:
                raise ValueError
            d[0] = kwargs.get('d')[0]
            d[1] = kwargs.get('d')[1]

            return self._first(x_new, d)
        # elif self.t == 2:
        #     return
        # elif self.t == 3:
        #     return

    cdef _first(self, np.ndarray[double, ndim=1] x_new, double[:] d):
        cdef double[:] r = <double[: self.n + 1]>malloc((self.n + 1) * sizeof(double))
        print(x_new.ndim)
        # prepare value container
        # a, b, h, alpha, beta, dy = prepare_empty_container_with_same_size(self.x.shape, 6)
        #
        # # default value
        # a[0], a[-1] = 0, np.nan
        # b[0], b[-1] = d[0], np.nan
        # # j = 1, 2, ..., n - 1
        # h[-1] = np.nan
        # # j = 2, 3, ..., n - 1
        # alpha[0], alpha[-1] = np.nan, np.nan
        # # j = 2, 3, ..., n - 1
        # beta[0], beta[-1] = np.nan, np.nan
        # # j = 2, 3, ..., n - 1
        # dy[0], dy[-1] = d[0], d[1]

    cdef _second(self):
        pass

    cdef _third(self):
        pass
