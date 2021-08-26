import numpy as np
import scipy as sp
import scipy.stats
from ._typing import array_like, Optional, Float, NoReturn, Boolean, number_check, array_cross_check, array_check

from .decorator import after_fit
from .utils import standardization, cov, prepare_empty_container_with_same_size, StatisticsModule
from .separation_time_space import EOFCustom


class SVD(StatisticsModule):
    """
    奇异值分解（singular value decomposition, SVD）通过对两变量场的
    交叉协方差矩阵运算，分解耦合场的时空场，获得其空间和时间的高相关信息区，
    以此来表征两变量之间的相互关系

    此方法区别于 np.linalg.svd ，该方法在其基础之上增加了时空分析，
    更适用于大气与海洋中科学问题的分析
    """
    def __init__(self, contribution: Float = 0.85) -> None:
        """
        :param contribution: 贡献率，可选，默认 85%
                用以评估所取模态的个数，其范围为 [0, 1]
        """
        super(SVD, self).__init__()

        self.contribution = number_check(contribution, max=1, min=0)

        # 原始数据
        self._x = None
        self._y = None

        # np.linalg.svd 计算后的左右奇异矩阵
        # 即为空间场
        self._u = None
        self._v = None

        # 时间场
        self._l = None
        self._r = None

        # SVD 特征值
        self._lambda = None

    def fit(self, x: array_like, y: array_like):
        """
        对已标准化的数据进行 SVD 分析
        :param x: array_like, 2d
            需要计算的左场（已标准化），维度为 (空间, 时间)
        :param y: array_like, 2d
            需要计算的右场（已标准化），维度为 (空间, 时间)
        :return: self
        """
        x, y = array_cross_check(x, y, dim=2)

        self._x = x
        self._y = y

        sxy = np.dot(x, y.T) / x.shape[1]
        u, _lambda, vt = np.linalg.svd(sxy)
        v = vt.T
        self._lambda = _lambda

        self._u = u / np.multiply(u, u).sum(axis=0)
        self._v = v / np.multiply(v, v).sum(axis=0)

        self._l = u.T @ x
        self._r = vt @ y

        return self

    def fit_transform(self, x: array_like, y: array_like):
        """
        对未标准化的数据进行 SVD 分析
        :param x: array_like, 2d
            需要计算的左场（未标准化），维度为 (空间, 时间)
        :param y: array_like, 2d
            需要计算的右场（未标准化），维度为 (空间, 时间)
        :return: self
        """
        x, y = array_cross_check(x, y, dim=2)

        # 对变量场 X 和 Y 进行标准化预处理
        x = standardization(x, axis=1)
        y = standardization(y, axis=1)

        return self.fit(x, y)

    @after_fit
    def k(self, k: Optional[int] = None) -> int:
        """
        由模型初始化时所赋予的贡献率自动确定提取模态的个数
        :return: int，模态个数
        """
        if k is None:
            k = np.count_nonzero((self._lambda ** 2 / np.sum(self._lambda ** 2)).cumsum() <= self.contribution) + 1
            return k if k <= self._lambda.size else self._lambda.size
        elif k >= self._lambda.size:
            return self._lambda.size
        else:
            return k

    @after_fit
    def u(self, k: Optional[int] = None) -> np.ndarray:
        return self._u[:, : self.k(k)]

    @after_fit
    def v(self, k: Optional[int] = None) -> np.ndarray:
        return self._v[:, : self.k(k)]

    @after_fit
    def var_ctrb(self, k: Optional[int] = None) -> np.ndarray:
        """
        方差贡献率
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        return (self._lambda**2 / np.sum(self._lambda**2))[: self.k(k)]

    @after_fit
    def cum_var_ctrb(self, k: Optional[int] = None) -> np.ndarray:
        """
        累积方差贡献率
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        return self.var_ctrb(k=self._lambda.size).cumsum()[: self.k(k)]

    @after_fit
    def t_left(self, k: Optional[int] = None) -> np.ndarray:
        """
        左时间模态
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        return self._l[: self.k(k), :]

    @after_fit
    def t_right(self, k: Optional[int] = None) -> np.ndarray:
        """
        右时间模态
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        return self._r[: self.k(k), :]

    @after_fit
    def v_left(self, k: Optional[int] = None) -> np.ndarray:
        """
        左空间模态
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        return self._u[: self.k(k), :]

    @after_fit
    def v_right(self, k: Optional[int] = None) -> np.ndarray:
        """
        右空间模态
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        return self._v[: self.k(k), :]

    @after_fit
    def tcc(self, k: Optional[int] = None) -> np.ndarray:
        """
        时间相关系数
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        k = self.k(k)
        c = np.empty(shape=self._lambda[: k].shape)
        for i in np.ndindex(c.shape):
            c[i] = (self._l[i, :] * self._r[i, :]).mean() / \
                   np.sqrt((self._l[i, :]**2).mean()) / np.sqrt((self._r[i, :]**2).mean())
        return c

    @after_fit
    def left_heteogeneous_tcc(self, k: Optional[int] = None) -> np.ndarray:
        """
        左时间异性相关系数
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        r = np.empty((self._x.shape[0], self.k(k)))

        it = np.nditer(r, flags=['multi_index'], op_flags=['readwrite'])
        with it:
            for x in it:
                i, j = it.multi_index
                x[...] = sp.stats.pearsonr(self._x[i, :], self._r[j, :])[0]

        return r

    @after_fit
    def right_heteogeneous_tcc(self, k: Optional[int] = None) -> np.ndarray:
        """
        右时间异性相关系数
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        r = np.empty((self._x.shape[0], self.k(k)))

        it = np.nditer(r, flags=['multi_index'], op_flags=['readwrite'])
        with it:
            for x in it:
                i, j = it.multi_index
                x[...] = sp.stats.pearsonr(self._y[i, :], self._l[j, :])[0]

        return r

    @after_fit
    def left_homogeneous_tcc(self, k: Optional[int] = None) -> np.ndarray:
        """
        左时间同性相关系数
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        r = np.empty((self._x.shape[0], self.k(k)))

        it = np.nditer(r, flags=['multi_index'], op_flags=['readwrite'])
        with it:
            for x in it:
                i, j = it.multi_index
                x[...] = sp.stats.pearsonr(self._x[i, :], self._l[j, :])[0]

        return r

    @after_fit
    def right_homogeneous_tcc(self, k: Optional[int] = None) -> np.ndarray:
        """
        右时间同性相关系数
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :return: ndarray
        """
        r = np.empty((self._x.shape[0], self.k(k)))

        it = np.nditer(r, flags=['multi_index'], op_flags=['readwrite'])
        with it:
            for x in it:
                i, j = it.multi_index
                x[...] = sp.stats.pearsonr(self._y[i, :], self._r[j, :])[0]

        return r

    @after_fit
    def monte_carlo_test(self, k: Optional[int] = None, alpha: Optional[float] = 0.05) -> np.ndarray:
        """
        蒙特卡洛检验
        :param k: 模态个数，可选，默认 None，将由模型自动决定
        :param alpha: 显著性水平
        :return: ndarray[float]，显著性
        """
        k = self.k(k)
        beta = np.empty(shape=(100, k))
        # 生成与原变量场大小一致的两个资料阵
        x = np.random.normal(0, 1, size=self._x.shape)
        y = np.random.normal(0, 1, size=self._y.shape)
        for epoch in np.arange(100):
            sxy = np.dot(x, y.T) / x.shape[1]
            # 仅需要特征值
            eta = np.linalg.svd(sxy, compute_uv=False)
            beta[epoch, :] = (eta[: k]**epoch) / (eta**epoch).sum()

        ctrb = (self._lambda / self._lambda.sum())[: k]
        number = np.zeros(shape=(k,))
        for i in np.ndindex(beta.shape[1]):
            data = beta[:, i].squeeze()
            # 单调递增且小于原变量方差贡献率
            number[i] = np.where(np.logical_and(
                np.append(np.asarray(True), np.diff(data, axis=0) > 0),
                data < ctrb[i])
            )[0].size

        return number >= 100 - np.around(alpha, 2) * 100


class CCA(StatisticsModule):
    """
    典型相关分析（canonical correlation analyses, CCA）
    通过寻求量要素变量场最佳的线性关系把两要素变量场变为多对典型变量，
    利用每对典型变量之间的相关性来反映两要素变量场的整体相关性
    """
    def __init__(self):
        super(CCA, self).__init__()

        # 原始变量场
        self._x = None
        self._y = None

        # 特征值
        self._lambda = None

        # 典型变量场（列向量）
        self._u = None
        self._v = None

        # 典型载荷特征向量（列向量）
        self._a = None
        self._b = None

    def fit(self, x: array_like, y: array_like):
        """
        对已经标准化的数据进行计算
        :param x: array_like
            需要计算的 x（已标准化），输入数组为 (特征, 样本)
        :param y: array_like
            需要计算的 y（已标准化），输入数组为 (特征, 样本)
        :return: self
        """
        x, y = array_cross_check(x, y, dim=2)

        self._x = x
        self._y = y

        # 求协方差
        sxx = cov(x, x)
        sxy = cov(x, y)
        syx = cov(y, x)
        syy = cov(y, y)

        # 求特征值与特征向量，获取 b
        s = np.linalg.inv(syy) @ syx @ np.linalg.inv(sxx) @ sxy

        _lambda, b = np.linalg.eig(s)

        # 降序排列
        condition = np.argsort(-_lambda)
        _lambda = _lambda[condition]
        b = b[:, condition]

        self._lambda = _lambda

        # 求 a，计算典型变量
        prefix = np.linalg.inv(sxx) @ sxy
        a = np.empty(shape=b.shape)
        u, v = prepare_empty_container_with_same_size((_lambda.size, x.shape[1]), 2)
        for i, in np.ndindex(_lambda.shape):
            a[:, i] = prefix @ b[:, i] / np.sqrt(_lambda[i])
            u[i] = a[:, i].T @ x
            v[i] = b[:, i].T @ y
        self._u = u
        self._v = v

        self._a = a
        self._b = b

        return self

    def fit_transform(self, x: array_like, y: array_like):
        """
        TODO: 说明维度 (时间, 空间)
        对未标准化的数据进行计算
        :param x: array_like
            需要计算的 x（未标准化），形状为 [n_feature, n_sample]
        :param y: array_like
            需要计算的 y（未标准化），形状为 [n_feature, n_sample]
        :return: self
        """
        x, y = array_cross_check(x, y, dim=2)
        # 对变量场 X 和 Y 进行标准化预处理
        x = standardization(x, axis=1)
        y = standardization(y, axis=1)

        return self.fit(x, y)

    @after_fit
    def r(self, k: Optional[int] = None):
        """
        典型相关系数
        """
        return np.sqrt(self._lambda) if k is None else np.sqrt(self._lambda)[: k]

    @after_fit
    def a(self, k: Optional[int] = None):
        """
        对应 x 的典型载荷特征向量
        """
        return self._a if k is None else self._a[:, : k]

    @after_fit
    def b(self, k: Optional[int] = None):
        """
        对应 y 的典型载荷特征向量
        """
        return self._b if k is None else self._b[:, : k]

    @after_fit
    def u(self, k: Optional[int] = None):
        """
        对应 x 的典型变量场
        """
        return self._u if k is None else self._u[: k, :]

    @after_fit
    def v(self, k: Optional[int] = None):
        """
        对应 y 的典型变量场
        """
        return self._v if k is None else self._v[: k, :]

    @after_fit
    def p(self, k: Optional[int] = None):
        """
        对应 x 的同类相关图
        """
        result = np.cov(self._x, rowvar=True) @ self._a
        return result if k is None else result[:, : k]

    @after_fit
    def q(self, k: Optional[int] = None):
        """
        对应 y 的同类相关图
        """
        result = np.cov(self._y, rowvar=True) @ self._b
        return result if k is None else result[:, : k]

    @after_fit
    def chi2_test(self, k: Optional[int] = None):
        """
        卡方检验
        """
        pass


class BPCCA(StatisticsModule):
    def __init__(self, contribution=0.85):

        super().__init__()
        self.contribution = contribution

        self.cca: CCA or None = None

    def fit(self, x: array_like, y: array_like, k: Optional[int] = None):
        # TODO: 说明数据输入维度为 (空间, 时间)
        x, y = array_cross_check(x, y, dim=2)

        # 第一步 -> 对变量进行 EOF 分解
        # TODO:
        eof_x = EOFCustom(contribution=self.contribution).fit_transform(x)
        eof_y = EOFCustom(contribution=self.contribution).fit_transform(y)

        x_t = eof_x.t(k)
        y_t = eof_y.t(k)
        eig_x = eof_x.lambda_(k)
        eig_y = eof_y.lambda_(k)

        x_t_standardization = x_t / np.sqrt(eig_x).reshape(-1, 1)
        y_t_standardization = y_t / np.sqrt(eig_y).reshape(-1, 1)

        self.cca = CCA().fit(x_t_standardization, y_t_standardization)

        return self

    @after_fit
    def r(self):
        return self.cca.r()

    @after_fit
    def a(self):
        return self.cca.a()

    @after_fit
    def b(self):
        return self.cca.b()

    @after_fit
    def u(self):
        return self.cca.u()

    @after_fit
    def v(self):
        return self.cca.v()
