from typing import Callable, List, Union
import numpy as np
import cvxpy as cp
from scipy.optimize import approx_fprime


class VMCON:
    def __init__(
        self,
        function: Callable,
        equality_constraints: List[Callable],
        inequality_constraints: List[Callable],
        n: int,
    ) -> None:
        self.f = function
        """The function, f, to be minimised (or maximises)"""

        self.equality_constraints = equality_constraints
        """The set of equality constraint equations (actaully a list in this implementation)
        
        Within the paper, this would be c_i where i=1,...,k
        """

        self.inequality_constraints = inequality_constraints
        """The set of inequality constraint equations (actaully a list in this implementation)
        
        Within the paper, this would be c_i where i=k+1,...,m
        """

        self.n = n
        """The dimensionality of the problem. f is a function with n inputs."""

        self.m = len(equality_constraints) + len(inequality_constraints)
        """The total number of constraints on this problem."""

        self.j = 0
        """An integer identifying the current iteration of VMCON"""

    def run_vmcon(
        self, x: Union[np.ndarray, None] = None, max_iter: int = 100, epsilon=1e-8
    ):
        """

        :param x: an initial estimate solution vector of f. Defaults to all 0.
        :type x: np.ndarray
        """
        if x is None:
            x = np.zeros((self.n,))

        B = np.identity(max(self.n, self.m))

        # These None's will flag to the linesearch that j = 1
        mu_equality = None
        mu_inequality = None

        for _ in range(max_iter):
            delta, lamda_equality, lamda_inequality = self.solve_qsp(x, B)

            # Flow chart on page 8 seems to suggest we check the criteria here
            # but surely that should be done after the line search is complete
            # and we find x^(j)??
            # NOTE: Maybe we should check at the top of the loop?
            # ... but then what if the last x is correct? Maybe add 1 to max iter?

            if self.convergence_test(
                x, delta, lamda_equality, lamda_inequality, epsilon
            ):
                break

            alpha, mu_equality, mu_inequality = self.perform_linesearch(
                mu_equality, mu_inequality, lamda_equality, lamda_inequality, delta, x
            )

            xj = x + alpha * delta

            B = self.calculate_new_B(B, x, xj, lamda_equality, lamda_inequality)

            x = xj

        else:
            raise Exception(
                f"Could not converge on a feasible solution after {max_iter} iterations."
            )

        return x, lamda_equality, lamda_inequality

    def convergence_test(
        self,
        x_jm1: np.ndarray,
        delta_j: np.ndarray,
        lamda_equality_i: np.ndarray,
        lamda_inequality_i: np.ndarray,
        epsilon: float,
    ) -> bool:
        abs_df_dot_delta = abs(np.dot(approx_fprime(x_jm1, self.f), delta_j))
        abs_equality__err = abs(
            np.sum(
                [
                    lamda * c(x_jm1)
                    for lamda, c in zip(lamda_equality_i, self.equality_constraints)
                ]
            )
        )
        abs_inequality__err = abs(
            np.sum(
                [
                    lamda * c(x_jm1)
                    for lamda, c in zip(lamda_inequality_i, self.inequality_constraints)
                ]
            )
        )

        return abs_df_dot_delta + abs_equality__err + abs_inequality__err < epsilon

    @staticmethod
    def _powells_gamma(gamma: np.ndarray, ksi: np.ndarray, B: np.ndarray):
        ksiTBksi = ksi.T @ B @ ksi  # used throughout eqn 10
        ksiTgamma = ksi.T @ gamma  # dito, to reduce amount of matmul

        theta = 1.0
        if ksiTgamma < 0.2 * ksiTBksi:
            theta = 0.8 * ksiTBksi / (ksiTBksi - ksiTgamma)

        return theta * gamma + (1 - theta) * (B @ ksi)  # eqn 9

    def calculate_new_B(
        self,
        B: np.ndarray,
        x_jm1: np.ndarray,
        x_j: np.ndarray,
        lamda_equality: np.ndarray,
        lamda_inequality: np.ndarray,
    ):
        # xi (the symbol name) would be a bit confusing in this context,
        # ksi is how its pronounced in modern greek
        # reshape ksi to be a matrix
        ksi = (x_j - x_jm1).reshape((2, 1))

        g1 = self.dL(x_j, lamda_equality, lamda_inequality)
        g2 = self.dL(x_jm1, lamda_equality, lamda_inequality)
        gamma = (g1 - g2).reshape((2, 1))

        gamma = self._powells_gamma(gamma, ksi, B)

        B = (
            B
            - ((B @ ksi @ ksi.T @ B) / (ksi.T @ B @ ksi))
            + ((gamma @ gamma.T) / (gamma.T @ ksi))
        )  # eqn 8

        return B

    def dL(self, x, lamda_equality, lamda_inequality):
        fprime = approx_fprime(x, self.f)
        c_equality_prime = sum(
            [
                lamda * approx_fprime(x, c)
                for lamda, c in zip(lamda_equality, self.equality_constraints)
            ]
        )
        c_inequality_prime = sum(
            [
                lamda * approx_fprime(x, c)
                for lamda, c in zip(lamda_inequality, self.inequality_constraints)
            ]
        )

        return fprime - c_equality_prime - c_inequality_prime

    @staticmethod
    def _sum_lamda_constraints(lamdas, constraints, x):
        return np.array([li * ci(x) for li, ci in zip(lamdas, constraints)]).sum()

    def L(self, x, lamda_equality, lamda_inequality):
        sum_equality_constraints = self._sum_lamda_constraints(
            lamda_equality, self.equality_constraints, x
        )
        sum_inequality_constraints = self._sum_lamda_constraints(
            lamda_inequality, self.inequality_constraints, x
        )

        return self.f(x) - sum_equality_constraints - sum_inequality_constraints

    @staticmethod
    def _calculate_mu_i(mu_im1: np.ndarray, lamda: np.ndarray):
        if mu_im1 is None:
            return np.abs(lamda)

        # element-wise maximum is assumed
        return np.maximum(np.abs(lamda), 0.5 * (mu_im1 + np.abs(lamda)))

    def perform_linesearch(
        self,
        mu_equality: Union[np.ndarray, None],
        mu_inequality: Union[np.ndarray, None],
        lamda_equality: np.ndarray,
        lamda_inequality: np.ndarray,
        delta: np.ndarray,
        x_jm1: np.ndarray,
    ):
        mu_equality = self._calculate_mu_i(mu_equality, lamda_equality)
        mu_inequality = self._calculate_mu_i(mu_inequality, lamda_inequality)

        def phi(x: np.ndarray):
            f = self.f(x)
            sum_equality = (
                mu_equality
                * np.abs(np.array([c(x) for c in self.equality_constraints]))
            ).sum()
            sum_inequality = (
                mu_inequality
                * np.abs(np.array([min(0, c(x)) for c in self.inequality_constraints]))
            ).sum()

            return f + sum_equality + sum_inequality

        def dphi(x: np.ndarray):
            return approx_fprime(x, phi)

        alpha = 1.0
        one = np.ones_like(x_jm1)
        prev_x = x_jm1
        for _ in range(10):
            x = x_jm1 + alpha * delta
            if phi(prev_x) - phi(x) < (0.1 * dphi(x)).all():
                break

            alpha = (alpha * dphi(x)) / 2 * (phi(x_jm1) - phi(one) - alpha * dphi(x))
            prev_x = x
        else:
            raise Exception("Line search did not converge on an approimate minima")

        return alpha, mu_equality, mu_inequality

    def solve_qsp(self, x: np.ndarray, B: np.ndarray):
        """
        Q(d) = f + dTf' + (1/2)dTBd

        f can be ignored here since it will have no effect on the minimisation
        of Q(d).

        The problem will be defined in its standard form:
            Q(d) = (1/2)dTBd + f'Td

        which, for the sake of continuity with the provided references, is:
            (1/2)xTPx + qTx
        where:
            x = d
            P = B
            q = f'

        The notation of the constraints on this QSP are as follows:
            - Gx <= h
            - Ax = b

        in this problem,
            - G = derivative of the inequality constraints
            (assumes all inequality constraints are <=) at x^(j-1)
            - h = the negative of the value of the constraints at x^(j-1)
            - A = derivative of the equality constraints at x^(j-1)
            - b = the negative value of the constraints at x^(j-1)

        References:
            - https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
            - https://www.cis.upenn.edu/~cis515/cis515-11-sl12.pdf (specifically comments
            on page 454 (page 8 of the provided pdf))
        """
        P = B
        fprime = approx_fprime(x, self.f)  # derivative of f at x^(j-1)
        q = fprime.T

        equality_constraint_values = np.array([c(x) for c in self.equality_constraints])
        inequality_constraint_values = np.array(
            [c(x) for c in self.inequality_constraints]
        )

        ceprime = np.array(
            [approx_fprime(x, c) for c in self.equality_constraints]
        )  # derivative of the equality constraints at x^(j-1)

        ciprime = np.array(
            [approx_fprime(x, c) for c in self.inequality_constraints]
        )  # derivative of the inequality constraints at x^(j-1)

        A = ceprime
        b = -equality_constraint_values

        G = -ciprime
        h = inequality_constraint_values

        delta = cp.Variable(x.shape)
        problem = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(delta, P) + q.T @ delta),
            [G @ delta <= h, A @ delta == b],
        )

        problem.solve()

        lamda_equality = -problem.constraints[1].dual_value
        lamda_inequality = problem.constraints[0].dual_value

        return delta.value, lamda_equality, lamda_inequality


if __name__ == "__main__":
    f = lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2
    ce = [lambda x: x[0] - (2 * x[1]) + 1]
    ci = [
        lambda x: ((x[0] ** 2) / 4) + (x[1] ** 2) - 1,
    ]

    vmcon = VMCON(f, ce, ci, 2)

    print(vmcon.run_vmcon(np.array([2.0, 2.0]), 10))
