from typing import Optional
import numpy as np


class VMCONConvergenceException(Exception):
    """Base class for an exception that indicates VMCON has
    failed to converge. This exception allows certain diagnostics
    to be passed and propogated with the exception."""

    def __init__(
        self,
        *args: object,
        x: Optional[np.ndarray] = None,
        lamda_equality: Optional[np.ndarray] = None,
        lamda_inequality: Optional[np.ndarray] = None
    ) -> None:
        """Constructor for the exception raised when VMCON cannot converge
        on a feasible solution.

        :param args: arguments passed to the Exception __init__() method. For example, an error message.
        :param x: the lastest `x` vector considered by the optimisation loop.
        :param lamda_equality: the latest Lagrange multipliers for equality constraints calculated.
        :param lamda_inequality: the latest Lagrange multipliers for equality constraints calculated.
        """
        super().__init__(*args)

        self.x = x
        self.lamda_equality = lamda_equality
        self.lamda_inequality = lamda_inequality


class _QspSolveException(Exception):
    """An exception that should only be used internally
    to identify that the QSP has failed to solve."""

    pass


class LineSearchConvergenceException(Exception):
    """Indicates the line search portion of VMCON was unable to
    converge on a solution within a pre-defined number of iterations"""

    pass
