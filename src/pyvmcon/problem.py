from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple, TypeVar

import numpy as np

T = TypeVar("T", np.ndarray, np.number, float)


class Result(NamedTuple):
    """The data from calling a problem"""

    f: T
    """Value of the objective function"""
    df: T
    """Derivative of the objective function"""
    eq: np.ndarray
    """1D array of the values of the equality constraints with shape"""
    deq: np.ndarray
    """2D array of the derivatives of the equality constraints wrt
    each component of `x`
    """
    ie: np.ndarray
    """1D array of the values of the inequality constraints"""
    die: np.ndarray
    """2D array of the derivatives of the inequality constraints wrt
    each component of `x`
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
    def __call__(self, x: np.ndarray) -> Result:
        pass

    @property
    @abstractmethod
    def num_equality(self) -> int:
        """Returns the number of equality constraints this problem has"""

    @property
    def has_equality(self) -> bool:
        return self.num_equality > 0

    @property
    def has_inequality(self) -> bool:
        return self.num_inequality > 0

    @property
    @abstractmethod
    def num_inequality(self) -> int:
        """Returns the number of inequality constraints this problem has"""

    @property
    def total_constraints(self) -> int:
        """Returns the total number of constraints `m`"""
        return self.num_equality + self.num_inequality


_FunctionAlias = Callable[[np.ndarray], T]
_DerivativeFunctionAlias = Callable[[np.ndarray], np.ndarray]


class Problem(AbstractProblem):
    """Provides a simple implementation of an AbstractProblem
    that essentially acts as a caller to equations to gather all
    of the various data.
    """

    def __init__(
        self,
        f: _FunctionAlias,
        df: _DerivativeFunctionAlias,
        equality_constraints: List[_FunctionAlias],
        inequality_constraints: List[_FunctionAlias],
        dequality_constraints: List[_DerivativeFunctionAlias],
        dinequality_constraints: List[_DerivativeFunctionAlias],
    ) -> None:
        super().__init__()

        self._f = f
        self._df = df
        self._equality_constraints = equality_constraints
        self._inequality_constraints = inequality_constraints
        self._dequality_constraints = dequality_constraints
        self._dinequality_constraints = dinequality_constraints

    def __call__(self, x: np.ndarray) -> Result:
        return Result(
            self._f(x),
            self._df(x),
            np.array([c(x) for c in self._equality_constraints]),
            np.array([c(x) for c in self._dequality_constraints]),
            np.array([c(x) for c in self._inequality_constraints]),
            np.array([c(x) for c in self._dinequality_constraints]),
        )

    @property
    def num_equality(self) -> int:
        return len(self._equality_constraints)

    @property
    def num_inequality(self) -> int:
        return len(self._inequality_constraints)
