"""Defines the interface for VMCON to exchange data with external software."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray

ScalarType = TypeVar("ScalarType", NDArray, np.number, float)
"""A scalar variable e.g. a single number (which could be a 0D numpy array)"""
VectorType = NDArray
"""A numpy array with only 1 dimension"""
MatrixType = NDArray
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
            cast("VectorType", np.array([c(x) for c in self.equality_constraints])),
            cast("MatrixType", np.array([c(x) for c in self.dequality_constraints])),
            cast("VectorType", np.array([c(x) for c in self.inequality_constraints])),
            cast("MatrixType", np.array([c(x) for c in self.dinequality_constraints])),
        )

    @property
    def num_equality(self) -> int:
        """The number of equality constraints this problem has."""
        return len(self.equality_constraints)

    @property
    def num_inequality(self) -> int:
        """The number of inequality constraints this problem has."""
        return len(self.inequality_constraints)
