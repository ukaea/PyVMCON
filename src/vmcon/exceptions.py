from typing import Optional
from .types import NumpyVector


class VMCONConvergenceException(Exception):
    def __init__(
        self,
        *args: object,
        x: Optional[NumpyVector] = None,
        lamda_equality: Optional[NumpyVector] = None,
        lamda_inequality: Optional[NumpyVector] = None
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


class LineSearchConvergenceException(Exception):
    pass
