"""
Clustering analysis
"""
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import kmeans, vq, whiten
from . import som
from ._typing import array_like, NoReturn


class KMeans:
    def __init__(self, a: array_like) -> None:
        self.a = np.asarray(a)

        self._label = None

    def __call__(self, category: int):

        a = whiten(self.a)
        id_ = kmeans(a, category)[0]
        self._label = vq(a, id_)[0]

        return self

    @property
    def label(self) -> np.array:
        if self._label is None:
            raise RuntimeError('Please execute this module first.')

        return self._label


class HierarchicalClustering:
    def __init__(self, a: array_like, metric: str = 'euclidean') -> NoReturn:
        """
        :param a: data with (n_sample, n_feature)
        :param metric : str or function, optional
            The distance metric to use. The distance function can
            be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
            'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
            'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
            'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
            'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
        """
        self._distance = sch.distance.pdist(np.asarray(a), metric)

        self._cluster = None

    def __call__(self, t: int or str = 0, method: str = 'average', **kwargs):
        z = sch.linkage(self._distance, method)
        self._cluster = sch.fcluster(z, t, **kwargs)

    @property
    def cluster(self) -> np.array:
        if self._cluster is None:
            raise RuntimeError('Please execute this module first.')

        return self._cluster


class SOM(som.Som):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 decay_function=som.asymptotic_decay,
                 neighborhood_function='gaussian', random_seed=None):
        super(SOM, self).__init__(
            x, y, input_len, sigma, learning_rate,
            decay_function=decay_function,
            neighborhood_function=neighborhood_function,
            random_seed=random_seed
        )
