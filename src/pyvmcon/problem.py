"""Defines the interface for VMCON to exchange data with external software."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ScalarType: TypeAlias = NDArray[np.floating] | np.floating | float
"""A scalar variable e.g. a single number (which could be a 0D numpy array)"""
VectorType: TypeAlias = NDArray[np.floating]
"""A numpy array with only 1 dimension"""
MatrixType: TypeAlias = VectorType
"""A numpy array with 2 dimensions"""


@dataclass
class Result:
    """The data from calling a problem."""

    f: ScalarType
    """Value of the objective function."""
    df: VectorType
    """Derivative of the objective function wrt to each input variable."""
    eq: VectorType
    """1D array of the values of the equality constraints with shape."""
    deq: MatrixType
    """2D array of the derivatives of the equality constraints wrt
    each input variable.
    """
    ie: VectorType
    """1D array of the values of the inequality constraints."""
    die: MatrixType
    """2D array of the derivatives of the inequality constraints wrt
    each input variable.
    """

    def __iter__(self) -> Iterator[ScalarType | VectorType | MatrixType]:
        """Convert the dataclass into an iterable.

        .. deprecated:: 2.4.2
                    Result is now a dataclass so should not be iterated over
                    (including tuple unpacking the object).
        """
        warnings.warn(
            "Using the Result class as a namedtuple is deprecated (e.g. by tuple "
            "unpacking the object). Instead you should access individual attributes of "
            "the dataclass.",
            DeprecationWarning,
            stacklevel=2,
        )
        return iter((self.f, self.df, self.eq, self.deq, self.ie, self.die))


class AbstractProblem(ABC):
    """A problem defines how VMCON gets data about your specific constrained system.

    The problem class is required to provide several properties and data gathering
    methods but the implementation of these is not prescribed.

    Note that when defining a problem, VMCON will **minimise** an objective function
    `f(x)` subject to some equality constraints `e(x) = 0` and some inequality
    constraints `i(x) >= 0`.
    """

    @abstractmethod
    def __call__(self, x: VectorType) -> Result:
        """Evaluate the optimisation problem at input x."""

    @property
    @abstractmethod
    def num_equality(self) -> int:
        """The number of equality constraints this problem has."""

    @property
    def has_equality(self) -> bool:
        """Indicates whether or not this problem has equality constraints."""
        return self.num_equality > 0

    @property
    def has_inequality(self) -> bool:
        """Indicates whether or not this problem has inequality constraints."""
        return self.num_inequality > 0

    @property
    @abstractmethod
    def num_inequality(self) -> int:
        """The number of inequality constraints this problem has."""

    @property
    def total_constraints(self) -> int:
        """The total number of constraints `m`."""
        return self.num_equality + self.num_inequality


_ScalarReturnFunctionAlias = Callable[[VectorType], ScalarType]
_VectorReturnFunctionAlias = Callable[[VectorType], VectorType]


@dataclass
class Problem(AbstractProblem):
    """A simple implementation of an AbstractProblem.

    It essentially acts as a caller to equations to gather all of the various data.

    Note that VMCON assumes minimisation of f and that inequality constraints are
    feasible when they return a value >= 0.
    """

    f: _ScalarReturnFunctionAlias
    df: _VectorReturnFunctionAlias
    equality_constraints: list[_ScalarReturnFunctionAlias] = field(default_factory=list)
    inequality_constraints: list[_ScalarReturnFunctionAlias] = field(
        default_factory=list
    )
    dequality_constraints: list[_VectorReturnFunctionAlias] = field(
        default_factory=list
    )
    dinequality_constraints: list[_VectorReturnFunctionAlias] = field(
        default_factory=list
    )

    def __call__(self, x: VectorType) -> Result:
        """Evaluate the problem at input point x."""
        return Result(
            self.f(x),
            self.df(x),
            np.array([c(x) for c in self.equality_constraints]),
            np.array([c(x) for c in self.dequality_constraints]),
            np.array([c(x) for c in self.inequality_constraints]),
            np.array([c(x) for c in self.dinequality_constraints]),
        )

    @property
    def num_equality(self) -> int:
        """The number of equality constraints this problem has."""
        return len(self.equality_constraints)

    @property
    def num_inequality(self) -> int:
        """The number of inequality constraints this problem has."""
        return len(self.inequality_constraints)

    def __iter__(
        self,
    ) -> Iterator[
        _ScalarReturnFunctionAlias
        | _VectorReturnFunctionAlias
        | list[_ScalarReturnFunctionAlias]
        | list[_VectorReturnFunctionAlias]
    ]:
        """Convert the dataclass into an iterable.

        .. deprecated:: 2.4.2
                    Problem is now a dataclass so should not be iterated over
                    (including tuple unpacking the object).
        """
        warnings.warn(
            "Using the Problem class as a namedtuple is deprecated (e.g. by tuple "
            "unpacking the object). Instead you should access individual attributes of "
            "the dataclass.",
            DeprecationWarning,
            stacklevel=2,
        )
        return iter((
            self.f,
            self.df,
            self.equality_constraints,
            self.inequality_constraints,
            self.dequality_constraints,
            self.dinequality_constraints,
        ))
