"""Exceptions and errors raised within VMCON."""

import numpy as np

from .problem import Result


class VMCONConvergenceException(Exception):
    """Base error for VMCON errors.

    This exception allows certain diagnostics to be passed and propagated
    with the exception.
    """

    def __init__(
        self,
        *args: object,
        x: np.ndarray | None = None,
        result: Result | None = None,
        lamda_equality: np.ndarray | None = None,
        lamda_inequality: np.ndarray | None = None,
    ) -> None:
        """Constructs an exception raised within VMCON.

        Parameters
        ----------
        args : List[object]
            arguments passed to the Exception __init__() method.
            For example, an error message.

        x : Optional[ndarray]
            The (j-1)th `x` vector.

        result : Optional[Result]
            The result of evaluating the (j-1)th `x`.

        lamda_equality : Optional[ndarray]
            The jth Lagrange multipliers for the equality constraints

        lamda_inequality : Optional[ndarray]
            The jth Lagrange multipliers for the inequality constraints

        """
        super().__init__(*args)

        self.x = x
        self.result = result
        self.lamda_equality = lamda_equality
        self.lamda_inequality = lamda_inequality


class _QspSolveException(Exception):
    """Internal exception indicating the QSP failed to solve."""


class QSPSolverException(VMCONConvergenceException):
    """Indicates VMCON failed to solve because the QSP Solver was unable to solve."""


class LineSearchConvergenceException(VMCONConvergenceException):
    """Indicates the line search was unable to converge."""
