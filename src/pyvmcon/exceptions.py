from typing import Optional

import numpy as np

from .problem import Result


class VMCONConvergenceException(Exception):
    """Base class for an exception that indicates VMCON has
    failed to converge. This exception allows certain diagnostics
    to be passed and propagated with the exception.
    """

    def __init__(
        self,
        *args: object,
        x: Optional[np.ndarray] = None,
        result: Optional[Result] = None,
        lamda_equality: Optional[np.ndarray] = None,
        lamda_inequality: Optional[np.ndarray] = None,
    ) -> None:
        """Constructor for the exception raised when VMCON cannot converge
        on a feasible solution.

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
    """An exception that should only be used internally
    to identify that the QSP has failed to solve.
    """


class QSPSolverException(VMCONConvergenceException):
    """Indicates VMCON failed to solve because the QSP Solver was unable
    to solve.
    """


class LineSearchConvergenceException(VMCONConvergenceException):
    """Indicates the line search portion of VMCON was unable to
    solve within a pre-defined number of iterations
    """
