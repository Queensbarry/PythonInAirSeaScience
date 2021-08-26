"""
Custom typing inherit from typing package.
"""
import numpy as np
from typing import Callable, List, NoReturn, TypeVar, Union, Tuple, Dict, Optional, Iterator

__all__ = [
    'Callable',
    'List',
    'NoReturn',
    'TypeVar',
    'Union',
    'Tuple',
    'Dict',
    'Optional',
    'Float',
    'Boolean',
    'Iterator',
    # Custom defined
    'T',
    'Number',
    'NumberOrArray',
    'array_like',
    # type checker
    'array_check',
    'array_cross_check',
    'int_check',
    'number_check'
]


T = TypeVar('T')
Number = Union[int, float]
NumberOrArray = Union[int, float, np.ndarray]

# Array like sequence.
array_like = TypeVar('array_like')

Float = float
Boolean = bool


# type checker goes following
def array_check(a: array_like, dim: Union[int, tuple] = None, shape: Optional[tuple] = None) -> np.ndarray:
    """
    Check array info.

    :param a: array_like
        origin array
    :param dim: int, tuple
        Expect dim with a, default None mean that not to check it.
    :param shape: optional, tuple
        Expect tuple with a, default None mean that not to check it.
    :return: np.ndarray
        Transfer `a` to a numpy array.
    """
    a = np.asarray(a)
    if isinstance(dim, int):
        dim = (dim,)

    if (dim is not None) and (a.ndim not in dim):
        raise ValueError(f'Input array dim must in {dim}, but got {a.ndim}.')

    if (shape is not None) and (a.shape != shape):
        raise ValueError(f'Expect shape is {shape}, but got {a.shape}.')

    return a


def array_cross_check(a: array_like, b: array_like, *, dim: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check two array have the same size

    :param a: array_like
        array like `a`
    :param b: array_like
        array like `b`
    :param dim: int
        array dim
    :return: np.ndarray, np.ndarray
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if (dim is None and a.ndim != b.ndim) or (dim is not None and a.ndim != b.ndim != dim):
        raise ValueError(f'Input array `a` and `b` must have same dim, but got {a.ndim}, {b.ndim}.')

    if a.shape != a.shape:
        raise ValueError(f'Input array must have same shape, but got {a.shape}, {b.shape}.')

    return a, b


def int_check(
        a: int or None, min_: Optional[int] = None, max_: Optional[int] = None,
        *, order: Optional[str] = None, nullable: Optional[bool] = False,
        chosen: Optional[tuple] = None) -> int or None:
    """
    Integer check.

    :param a: int
    :param min_: optional, int
        minimum value (include)
    :param max_: optional, int
        maximum value (include)
    :param order: optional, str
        check `a` is odd while `order` is `odd`
        check `a` is even while `order` is `even`
    :param nullable: optional, bool
        defined `a` can be null or not
    :param chosen: optional, tuple
        chosen of `a`
    :return: int
        Value has been checked.
    """
    if (not nullable) and (a is None):
        raise ValueError('Input value can not be `None`.')
    elif nullable and (a is None):
        return a

    if not isinstance(a, (int, np.int32, np.int64)):
        # Ignored this warning, editor wrong warning.
        raise ValueError(f'Input value must be int, but got {a.__class__.__name__}')

    if (min_ is not None) and (a < min_):
        raise ValueError(f'Input value can not small than {min_}.')

    if (max_ is not None) and (a > max_):
        raise ValueError(f'Input value can not bigger than {max_}.')

    if (order == 'odd') and (a % 2 != 1):
        raise ValueError('Input value must be odd.')
    elif (order == 'even') and (a % 2 != 1):
        raise ValueError('Input value must be even.')

    if (chosen is not None) and (a not in chosen):
        raise ValueError(f'Input value must in {chosen}')

    return a


def number_check(a: Number, *, min: Optional[int] = np.inf, max: Optional[int] = np.inf, include: Iterator[Boolean] = None) -> Number:
    """
    Number check

    :param a: Number
    :param min: optional, int
        minimum value (include)
    :param max: optional, int
        maximum value (include)
    :param include: optional

    :return: Number
        Value has been checked.
    """
    if include is None:
        include = [True, True]
    elif isinstance(include, Iterator):
        raise ValueError('Param `include` is not iterator.')
    elif tuple(include) != 2:
        raise ValueError('Param `include` length must be 2.')
    elif tuple(filter(lambda x: isinstance(x, bool), include)) != 2:
        raise ValueError('Param `include` element must be instance of `bool`.')

    if a is None:
        raise ValueError('Input value can not be `None`.')

    if not isinstance(a, (int, float)):
        raise ValueError(f'Input value must be number, but got {a.__class__.__name__}')

    # TODO
    if (min is not None) and (a < min):
        raise ValueError(f'Input value can not small than {min}.')

    if (max is not None) and (a > max):
        raise ValueError(f'Input value can not bigger than {max}.')

    return a


def string_check(a: str, options: tuple = None) -> str:
    """
    String check

    :param a: str
        origin string
    :param options: tuple
        options tuple
    :return: str
    """
    if not isinstance(a, str):
        raise ValueError('Input type must be `str`.')

    if (options is not None) and (a not in options):
        raise ValueError(f'Input must be one of {options}')

    return a
