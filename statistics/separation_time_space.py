import numpy as np
from dataclasses import dataclass
from scipy import fftpack
import numpy.linalg
import xarray as xr
from sklearn.linear_model import LinearRegression
from eofs.xarray import Eof

import scipy as sp
import scipy.linalg

from .decorator import after_fit
from .utils import StatisticsModule, standardization, cov
from ._typing import array_check, array_like, number_check, string_check, Optional
from .mixin import PreHandlerMixin


class EOFCustom(StatisticsModule):
    """
    EOF method.
    """
    def __init__(self, contribution: float = 0.85):

        super().__init__()

        self.contribution = contribution

        self._v = None
        self._t = None
        self._lambda = None

    def fit(self, x: array_like):
        """
        TODO: 说明输入数组是 (空间, 时间)
        :param x:
        :return:
        """
        x = np.asarray(x)

        if x.shape[0] > x.shape[1]:
            # 空间维度大于时间维度
            s = x.T @ x
            eig, vr = np.linalg.eig(s)

            sort_index = np.argsort(-eig)
            eig = eig[sort_index]
            vr = vr[:, sort_index]

            v = (x @ vr) / np.sqrt(eig)
            v = v / np.multiply(v, v).sum(axis=0)
        else:
            s = x @ x.T
            eig, v = np.linalg.eig(s)
            sort_index = np.argsort(-eig)
            eig = eig[sort_index]
            v = v[:, sort_index] / np.sqrt(np.multiply(v, v).sum(axis=0))

        self._v = v
        self._t = v.T @ x
        self._lambda = eig

        return self

    def fit_transform(self, x: array_like):
        """
        TODO: 说明输入数组是 (空间, 时间)
        :param x:
        :return:
        """
        x = np.asarray(x)
        return self.fit(standardization(x, axis=1))

    @after_fit
    def k(self, k: Optional[int] = None) -> int:
        if k is None:
            k = np.count_nonzero((self._lambda**2 / np.sum(self._lambda**2)).cumsum() <= self.contribution) + 1
            return k if k <= self._lambda.size else self._lambda.size
        elif k >= self._lambda.size:
            return self._lambda.size
        else:
            return k

    @after_fit
    def lambda_(self, k: Optional[int] = None):
        return self._lambda[: self.k(k)]

    @after_fit
    def var_ctrb(self, k: Optional[int] = None) -> np.ndarray:
        """
        方差贡献率
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        return (self._lambda / self._lambda.sum())[: self.k(k)]

    @after_fit
    def cum_vat_ctrb(self, k: Optional[int] = None) -> np.ndarray:
        """
        累积方差贡献率
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        return self.var_ctrb(k=self._lambda.size).cumsum()[: self.k(k)]

    @after_fit
    def v(self, k: Optional[int] = None):
        return self._v[:, :self.k(k)]

    @after_fit
    def t(self, k: Optional[int] = None):
        return self._t[:self.k(k), :]


class EofModify(Eof):
    def pcs(self, pcscaling=0, npcs=None):
        pcs = self._solver.pcs(pcscaling, npcs)
        pcdim = xr.IndexVariable('mode', range(pcs.shape[1]),
                                 attrs={'long_name': 'eof_mode_number'})
        coords = [self._time, pcdim]
        pcs = xr.DataArray(pcs, coords=coords, name='pcs')
        pcs.coords.update({coord.name: (coord.dims, coord.data)
                           for coord in self._time_ndcoords})
        return pcs


class POP(StatisticsModule):

    def __init__(self):

        super().__init__()

        sst = xr.open_dataset('F:/HadISST_sst.nc')['sst']
        sst = sst.sel(
            time=slice('1968-01-01', '2018-12-31'),
            latitude=slice(30, -30),
            longitude=slice(0, 120)
        )[::, ::-1, ::]
        sst_ano = (sst.groupby('time.month') - sst.groupby('time.month').mean())

        sst_ano_de = xr.apply_ufunc(self.de_apply, sst_ano)

        sst_ano_de_mean = sst_ano_de.mean('time')
        sst_ano_de_std = sst_ano_de.std('time')
        sst_ano_de_stand = xr.apply_ufunc(lambda x, m, s: (x - m) / s, sst_ano_de, sst_ano_de_mean, sst_ano_de_std)

        solver = EofModify(sst_ano_de_stand)
        eofs = solver.eofs(neofs=10, eofscaling=0)

        pcs = solver.pcs(npcs=10, pcscaling=0)
        pcs = np.loadtxt('F:/pcs.txt').reshape(10, 612).T

        self.pop(pcs, 1, eofs)

    def pop(self, pcs, k, eofs):
        cov0 = np.cov(pcs, rowvar=False)
        # cov1 = np.zeros((len(pcs.mode), len(pcs.mode)))
        cov1 = np.zeros((10, 10))

        for i in range(10):
            cov1[i, :] = np.cov(pcs.T[i, : -1], pcs.T[:, 1:])[0, 1:]

        cov0_inv = np.matrix(cov0).I
        A = np.matmul(cov1, cov0_inv)
        # ev, lr = np.linalg.eig(A)
        ev, lr = sp.linalg.eig(A, left=True, right=False)
        pi = lr.imag[:, k].squeeze()
        pr = lr.real[:, k].squeeze()

        z_l = np.matrix([[pr.dot(pr), pr.dot(pi)],
                         [pr.dot(pi), pi.dot(pi)]]).I
        z_c = np.matrix([pr.tolist(),
                         pi.tolist()])
        z_r = pcs.T
        z = np.matmul(z_l, z_c)
        z = np.matmul(z, z_r).getA()
        z[1] = z[1] * -1

        # 标准化
        z = np.apply_along_axis(lambda x: x - x.mean(), 1, z)
        mean = np.mean(z, 1)
        stddev = np.std(z, 1)
        z = np.apply_along_axis(lambda x: (x - x.mean()) / x.std(), 1, z)

        for i in range(10):
            zr = pr[:, i] * eofs.values[i, :, :]
            zi = pi[:, i] * eofs.values[i, :, :]
        # zr = (eofs.values * pr).sum(-1).swapaxes(0, -1)
        # zi = (eofs.values * pi).sum(-1).swapaxes(0, -1)
        # Z = np.array([zr * stddev[0], -zi * stddev[1]])

        print()

    @staticmethod
    def _detrend(data):
        X = np.arange(0, len(data)).reshape(-1, 1)
        Y = data
        X_ = X[~np.isnan(Y)]
        Y_ = data[~np.isnan(Y)]
        if X_.shape[0] != 0:
            reg = LinearRegression().fit(X_, Y_)
            Y_pre = reg.predict(X)
            return Y - Y_pre
        else:
            return np.full_like(data, np.nan)

    @classmethod
    def de_apply(cls, data):
        data_cal = data.reshape((data.shape[0], -1))
        data_de = np.apply_along_axis(cls._detrend, 0, data_cal)
        data_de = data_de.reshape(data.shape)
        return data_de


class PrincipalOscillationPatternAnalysis(PreHandlerMixin):

    def __init__(self, a: array_like, bias: bool = True, contribution: float = 0.85):

        self.x = array_check(a, 2)
        self._bias = bias

        self.eof = EOFCustom(a, bias, contribution)

    def __call__(self):

        # TODO: 增加公式

        # 第一步 -> 使用 EOF 取原资料场前 k 维个模态
        self.eof()
        k = self.eof.k
        x = self.eof.t(k)

        # 第二步 -> 计算回归系数矩阵 A
        # 无滞后协方差 C^{hat}_0
        c0 = np.cov(x, bias=self._bias)
        # 滞后步长为 1 的协方差 C^{hat}_1
        c1 = np.empty(c0.shape)
        for i in np.arange(c1.shape[0]):
            for j in np.arange(c1.shape[1]):
                _temp = 0
                for _t in np.arange(c0.shape[0] - 1):
                    _temp += x[i, _t + 1] * x[j, _t]
                _temp = _temp / (c0.shape[0] - 1)
                c1[i, j] = _temp
        a = np.dot(c1, np.linalg.inv(c1))

        # 第三步 -> 求 A 的特征向量 \lambda 、共轭特征值 \lambda_{*} 和对应的特征向量 V 及其共轭向量 V_{*}
        lambda_, v = np.linalg.eig(a)
        lambda_star, v_star = np.conjugate(lambda_), np.conjugate(v)

        # 第四步 -> 利用递推公式求出时间系数 Z(t)  / 或 Z_{Re} 和 Z_{Im}
        rho = np.abs(lambda_)
        omicron = np.angle(lambda_)
        z_t = np.fromfunction(lambda _i: rho**(_i + 1) + omicron * 1j, shape=lambda_.shape)

        # 第五步 -> 计算每对特征向量占总方差的百分比

        # 第六步 -> 计算振荡成分 P(t) 的振荡周期和振荡衰减时间
        v_re = np.real(v)
        v_im = np.imag(v)
        p_t = np.empty(shape=v.shape, dtype=np.complex)
        for j in np.arange(p_t.shape[1]):
            p_t[:, j] = 2 * rho**(j + 1) * np.cos(omicron[j] * j) * v_re[:, j] \
                        - (2 * rho**(j + 1) * np.sin(omicron[j] * j) * v_im[:, j]) * 1j
        t = 2 * np.pi / np.abs(omicron)
        tau = -1 / np.log(rho)

        # 第七步：用主成分的振荡型表示出原数据地理空间中气象变量的一个振荡成分

