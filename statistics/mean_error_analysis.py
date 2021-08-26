import numpy as np


class MeanErrorAnalysis:

    @staticmethod
    def mean_error(f: np.array, o: np.array) -> float:
        return (f - o).sum() / f.size

    @staticmethod
    def mean_absolute_error(f: np.array, o: np.array) -> float:
        return np.abs(f - o).sum() / f.size

    @staticmethod
    def relative_absolute_error(f: np.array, o: np.array) -> float:
        return np.abs((f - o) / o).sum() / f.size

    @staticmethod
    def root_mean_squared_error(f: np.array, o: np.array) -> float:
        return np.sqrt(np.power(f - o, 2).sum() / f.size)
