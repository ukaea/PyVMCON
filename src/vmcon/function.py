from abc import ABC, abstractmethod
from typing import Callable, Union
import numpy as np
import numpy.typing as npt
from scipy.optimize import approx_fprime

from .types import NumpyVector


class AbstractFunction(ABC):
    """Provides a common interface for classes
    that will wrap a function."""

    def __call__(self, x: NumpyVector) -> Union[npt.NDArray, np.number]:
        return self.f(x)

    @abstractmethod
    def f(self, x: NumpyVector, /) -> Union[npt.NDArray, np.number]:
        """The function that this building block represents.

        `f` takes a vector (even if said vector is 0-d) that is then transformed into a single output.
        This output can be a vector or scalar (this time a scalar does not need to be a 0-d vector).

        :param x: a vector of inputs to `f`

        :returns: a scalar (or a numpy array of shape (1,) or (1, 1)) that is the output of `f`.
        """
        pass

    def derivative(
        self,
        x: NumpyVector,
        /,
        *,
        epsilon: Union[float, None] = None,
    ) -> NumpyVector:
        """Calculate the derivative of `f` at `x`.

        Unless overidden this method uses `scipy.optimize.approx_fprime`
        to calculate an approximation of the derivative of `f` at `x`.

        :param x: the point in the domain to calculate the derivative of `f`.
        :param epsilon: the small step taken in `x` by `scipy.optimize.approx_fprime` (default: None, use the default epsilon from the scipy function).

        :returns: the derivative of `f` at `x`.
        """

        if epsilon is None:
            return approx_fprime(x, self.f)
        return approx_fprime(x, self.f, epsilon=epsilon)


class Function(AbstractFunction):
    """Provides a concrete implementation of a Function
    which can be built by passing `lambda`s to the constructor
    """

    def __init__(
        self,
        f: Callable[[npt.NDArray], Union[npt.NDArray, np.number]],
        df: Union[Callable[[npt.NDArray], Union[npt.NDArray, np.number]], None] = None,
    ) -> None:
        super().__init__()

        self._f = f
        self._df = df

    def f(self, x: NumpyVector, /) -> Union[npt.NDArray, np.number]:
        return self._f(x)

    def derivative(
        self, /, x: NumpyVector, *, epsilon: Union[float, None] = None
    ) -> NumpyVector:
        if self._df is None:
            return super().derivative(x, epsilon=epsilon)

        return self._df(x)
