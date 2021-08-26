from .average import depature
from .utils import normalization, standardization


class PreHandlerMixin:
    """
    Model pre-handler include depature, normalization and standardization.
    """
    depature = staticmethod(depature)
    normalization = staticmethod(normalization)
    standardization = staticmethod(standardization)
