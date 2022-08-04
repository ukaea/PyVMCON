from typing import List, Union
import numpy as np
import numpy.typing as npt
import cvxpy as cp
from scipy.optimize import approx_fprime

from .exceptions import VMCONConvergenceException, LineSearchConvergenceException
from .function import Function
from .types import Vector, NumpyVector, coerce_vector


def solve(
    f: Function,
    equality_constraints: List[Function],
    inequality_constraints: List[Function],
    x: Vector,
    max_iter: int = 10,
    epsilon: float = 1e-8,
):
    """The main solving loop of the VMCON non-linear constrained optimiser."""

    x = coerce_vector(x)

    # n is denoted in the VMCON paper
    # as the number of inputs the function
    # and the constraints take
    n = x.shape[0]

    # m is the total number of constraints
    m = len(equality_constraints) + len(inequality_constraints)

    # The paper uses the B matrix as the
    # running approximation of the Hessian
    B = np.identity(max(n, m))

    # These two values being None allows the line
    # search to realise that it is the first iteration
    mu_equality = None
    mu_inequality = None

    # Ensure these variables are visible to the exception
    lamda_equality = None
    lamda_inequality = None

    for _ in range(max_iter):
        # solve the quadratic subproblem to identify
        # our search direction and the Lagrange multipliers
        # for our constraints
        delta, lamda_equality, lamda_inequality = solve_qsp(
            f, equality_constraints, inequality_constraints, x, B
        )

        # Exit to optimisation loop if the convergence
        # criteria is met
        if convergence_test(
            f,
            equality_constraints,
            inequality_constraints,
            x,
            delta,
            lamda_equality,
            lamda_inequality,
            epsilon,
        ):
            break

        # perform a linesearch along the search direction
        # to mitigate the impact of poor starting conditions.
        alpha, mu_equality, mu_inequality = perform_linesearch(
            f,
            equality_constraints,
            inequality_constraints,
            mu_equality,
            mu_inequality,
            lamda_equality,
            lamda_inequality,
            delta,
            x,
        )

        # use alpha found during the linesearch to find xj.
        # Notice that the revision of matrix B needs the x^(j-1)
        # so our running x is not overriden yet!
        xj = x + alpha * delta

        # Revise matrix B
        B = calculate_new_B(
            f,
            equality_constraints,
            inequality_constraints,
            B,
            x,
            xj,
            lamda_equality,
            lamda_inequality,
        )

        # Update x for our next VMCON iteration
        x = xj

    else:
        raise VMCONConvergenceException(
            f"Could not converge on a feasible solution after {max_iter} iterations.",
            x=x,
            lamda_equality=lamda_equality,
            lamda_inequality=lamda_inequality,
        )

    return x, lamda_equality, lamda_inequality


def solve_qsp(
    f: Function,
    equality_constraints: List[Function],
    inequality_constraints: List[Function],
    x: NumpyVector,
    B: npt.NDArray,
):
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
    q = f.derivative(x).T

    equality_constraint_values = np.array([c(x) for c in equality_constraints])
    inequality_constraint_values = np.array([c(x) for c in inequality_constraints])

    ceprime = np.array(
        [c.derivative(x) for c in equality_constraints]
    )  # derivative of the equality constraints at x^(j-1)

    ciprime = np.array(
        [c.derivative(x) for c in inequality_constraints]
    )  # derivative of the inequality constraints at x^(j-1)

    A = ceprime
    b = -equality_constraint_values

    G = -ciprime
    h = inequality_constraint_values

    delta = cp.Variable(x.shape)
    problem_statement = cp.Minimize(0.5 * cp.quad_form(delta, P) + q.T @ delta)

    lamda_equality = np.array([])
    lamda_inequality = np.array([])

    if inequality_constraints and equality_constraints:
        problem = cp.Problem(
            problem_statement,
            [G @ delta <= h, A @ delta == b],
        )

        problem.solve()

        lamda_inequality = problem.constraints[0].dual_value
        lamda_equality = -problem.constraints[1].dual_value

    elif inequality_constraints and not equality_constraints:
        problem = cp.Problem(
            problem_statement,
            [G @ delta <= h],
        )

        problem.solve()

        lamda_inequality = problem.constraints[0].dual_value

    elif not inequality_constraints and equality_constraints:
        problem = cp.Problem(
            problem_statement,
            [A @ delta == b],
        )

        problem.solve()

        lamda_equality = -problem.constraints[0].dual_value

    else:
        problem = cp.Problem(problem_statement)

        problem.solve()

    return delta.value, lamda_equality, lamda_inequality


def convergence_test(
    f: Function,
    equality_constraints: List[Function],
    inequality_constraints: List[Function],
    x_jm1: NumpyVector,
    delta_j: NumpyVector,
    lamda_equality_i: NumpyVector,
    lamda_inequality_i: NumpyVector,
    epsilon: float,
) -> bool:
    abs_df_dot_delta = abs(np.dot(f.derivative(x_jm1), delta_j))
    abs_equality__err = abs(
        np.sum(
            [
                lamda * c(x_jm1)
                for lamda, c in zip(lamda_equality_i, equality_constraints)
            ]
        )
    )
    abs_inequality__err = abs(
        np.sum(
            [
                lamda * c(x_jm1)
                for lamda, c in zip(lamda_inequality_i, inequality_constraints)
            ]
        )
    )

    return abs_df_dot_delta + abs_equality__err + abs_inequality__err < epsilon


def _calculate_mu_i(mu_im1: Union[NumpyVector, None], lamda: NumpyVector):
    if mu_im1 is None:
        return np.abs(lamda)

    # element-wise maximum is assumed
    return np.maximum(np.abs(lamda), 0.5 * (mu_im1 + np.abs(lamda)))


def perform_linesearch(
    f: Function,
    equality_constraints: List[Function],
    inequality_constraints: List[Function],
    mu_equality: Union[NumpyVector, None],
    mu_inequality: Union[NumpyVector, None],
    lamda_equality: NumpyVector,
    lamda_inequality: NumpyVector,
    delta: NumpyVector,
    x_jm1: NumpyVector,
):
    mu_equality = _calculate_mu_i(mu_equality, lamda_equality)
    mu_inequality = _calculate_mu_i(mu_inequality, lamda_inequality)

    # TODO: Cache this function to avoid repeated calls to the objective (and constraints)
    def phi(alpha: np.floating):
        x = x_jm1 + alpha * delta
        sum_equality = (
            mu_equality * np.abs(np.array([c(x) for c in equality_constraints]))
        ).sum()
        sum_inequality = (
            mu_inequality
            * np.abs(np.array([min(0, c(x)) for c in inequality_constraints]))
        ).sum()

        return f(x) + sum_equality + sum_inequality

    # dphi(0) for unconstrained minimisation = F'(x_jm1)*delta
    # this is extended to constrained minimisation subtracting the
    # weighted constraints at 0.
    capital_delta = (approx_fprime(x_jm1, f) * delta).sum() - phi(0) + f(x_jm1)

    alpha = 1.0
    for _ in range(100):
        # exit if we satisfy the Armijo condition
        if phi(alpha) <= phi(0) + 0.1 * alpha * capital_delta:
            break

        alpha = max(
            0.1 * alpha,
            -0.5
            * alpha**2
            * capital_delta
            / (phi(alpha) - phi(0) - (alpha * capital_delta)),
        )

    else:
        raise LineSearchConvergenceException(
            "Line search did not converge on an approimate minima"
        )

    return alpha, mu_equality, mu_inequality


def _derivative_lagrangian(
    f: Function,
    equality_constraints: List[Function],
    inequality_constraints: List[Function],
    x: NumpyVector,
    lamda_equality: NumpyVector,
    lamda_inequality: NumpyVector,
):
    c_equality_prime = sum(
        [
            lamda * c.derivative(x)
            for lamda, c in zip(lamda_equality, equality_constraints)
        ]
    )
    c_inequality_prime = sum(
        [
            lamda * c.derivative(x)
            for lamda, c in zip(lamda_inequality, inequality_constraints)
        ]
    )

    return f.derivative(x) - c_equality_prime - c_inequality_prime


def _powells_gamma(gamma: np.ndarray, ksi: np.ndarray, B: np.ndarray):
    ksiTBksi = ksi.T @ B @ ksi  # used throughout eqn 10
    ksiTgamma = ksi.T @ gamma  # dito, to reduce amount of matmul

    theta = 1.0
    if ksiTgamma < 0.2 * ksiTBksi:
        theta = 0.8 * ksiTBksi / (ksiTBksi - ksiTgamma)

    return theta * gamma + (1 - theta) * (B @ ksi)  # eqn 9


def calculate_new_B(
    f: Function,
    equality_constraints: List[Function],
    inequality_constraints: List[Function],
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

    g1 = _derivative_lagrangian(
        f,
        equality_constraints,
        inequality_constraints,
        x_j,
        lamda_equality,
        lamda_inequality,
    )
    g2 = _derivative_lagrangian(
        f,
        equality_constraints,
        inequality_constraints,
        x_jm1,
        lamda_equality,
        lamda_inequality,
    )
    gamma = (g1 - g2).reshape((2, 1))

    gamma = _powells_gamma(gamma, ksi, B)

    B = (
        B
        - ((B @ ksi @ ksi.T @ B) / (ksi.T @ B @ ksi))
        + ((gamma @ gamma.T) / (gamma.T @ ksi))
    )  # eqn 8

    return B
