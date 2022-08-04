from typing import Callable

from numpy import floating
from .types import NumpyVector


def approximate_gradient(
    f: Callable[[NumpyVector], NumpyVector],
    x: NumpyVector,
    *,
    epsilon: floating = 1.4901161193847656e-08
):
    return (f(x + epsilon / 2) - f(x - epsilon / 2)) / epsilon
