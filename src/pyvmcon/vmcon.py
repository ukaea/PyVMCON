from typing import Union
import logging
import numpy as np
import cvxpy as cp

from .exceptions import (
    VMCONConvergenceException,
    LineSearchConvergenceException,
    _QspSolveException,
    QSPSolverException,
)
from .problem import AbstractProblem, Result

logger = logging.getLogger(__name__)
s_handler = logging.StreamHandler()
s_handler.setLevel(logging.WARNING)


def solve(
    problem: AbstractProblem,
    x: np.ndarray,
    lbs: np.ndarray = None,
    ubs: np.ndarray = None,
    *,
    max_iter: int = 10,
    epsilon: float = 1e-8,
    initial_B: np.ndarray = None,
):
    """The main solving loop of the VMCON non-linear constrained optimiser.

    Parameters
    ----------
    problem : AbstractProblem
        Defines the system to be minimised

    x : ndarray
        The initial starting `x` of VMCON

    lbs : ndarray
        Lower bounds of `x`. If `None`, no lower bounds are applied

    ubs : ndarray
        Upper bounds of `x`. If `None`, no upper bounds are applied

    max_iter : int
        The maximum iterations of VMCON before an exception is raised

    epsilon : float
        The tolerance used to test if VMCON has converged

    initial_B : ndarray
        Initial estimate of the Hessian matrix `B`. If `None`, `B` is the
        identity matrix of shape `(max(n,m), max(n,m))`.
    """

    if len(x.shape) != 1:
        raise ValueError(
            "Input vector `x` is not a 1D array or an nD array with only 1 non-singleton dimension"
        )

    if lbs is not None and (x < lbs).any():
        logger.warning(
            f"""x is initially in an infeasible region because at least one x is lower than a lower bound:
            {x - lbs = }"""
        )

    if ubs is not None and (x > ubs).any():
        logger.warning(
            f"""x is initially in an infeasible region because at least one x is greater than an upper bound
            {ubs - x = }"""
        )

    # n is denoted in the VMCON paper
    # as the number of inputs the function
    # and the constraints take
    n = x.shape[0]

    # m is the total number of constraints
    m = problem.total_constraints

    # The paper uses the B matrix as the
    # running approximation of the Hessian
    if initial_B is None:
        B = np.identity(max(n, m))
    else:
        B = initial_B

    # These two values being None allows the line
    # search to realise that it is the first iteration
    mu_equality = None
    mu_inequality = None

    # Ensure these variables are visible to the exception
    lamda_equality = None
    lamda_inequality = None

    for i in range(max_iter):
        print(f"Iteration {i}")
        result = problem(x)

        # solve the quadratic subproblem to identify
        # our search direction and the Lagrange multipliers
        # for our constraints
        try:
            delta, lamda_equality, lamda_inequality = solve_qsp(
                problem, result, x, B, lbs, ubs
            )
        except _QspSolveException as e:
            raise QSPSolverException(
                f"QSP failed to solve, indicating no feasible solution could be found.",
                x=x,
                result=result,
                lamda_equality=lamda_equality,
                lamda_inequality=lamda_inequality,
            ) from e

        # Exit to optimisation loop if the convergence
        # criteria is met
        if convergence_test(
            result,
            delta,
            lamda_equality,
            lamda_inequality,
            epsilon,
        ):
            break

        # perform a linesearch along the search direction
        # to mitigate the impact of poor starting conditions.
        alpha, mu_equality, mu_inequality = perform_linesearch(
            problem,
            result,
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
            problem,
            result,
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
            result=result,
            lamda_equality=lamda_equality,
            lamda_inequality=lamda_inequality,
        )

    return x, lamda_equality, lamda_inequality, result


def solve_qsp(
    problem: AbstractProblem,
    result: Result,
    x: np.ndarray,
    B: np.ndarray,
    lbs: np.ndarray,
    ubs: np.ndarray,
    tolerance=1e-4,
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
    delta = cp.Variable(x.shape)
    problem_statement = cp.Minimize(
        result.f + (1 / 2) * cp.quad_form(delta, B) + (result.df.T @ delta)
    )

    equality_index = 0

    constraints = []
    if problem.has_inequality:
        equality_index += 1
        constraints.append((result.die @ delta) + result.ie >= 0)
    if lbs is not None:
        equality_index += 1
        constraints.append(x + delta >= lbs)
    if ubs is not None:
        equality_index += 1
        constraints.append(x + delta <= ubs)
    if problem.has_equality:
        constraints.append((result.deq @ delta) + result.eq == 0)

    qsp = cp.Problem(problem_statement, constraints or None)
    qsp.solve(verbose=True, solver="OSQP", eps_rel=tolerance)

    if delta.value is None:
        raise _QspSolveException(f"QSP failed to solve: {qsp.status}")

    lamda_equality = np.array([])
    lamda_inequality = np.array([])

    if problem.has_inequality and problem.has_equality:
        lamda_inequality = qsp.constraints[0].dual_value
        lamda_equality = -qsp.constraints[equality_index].dual_value

    elif problem.has_inequality and not problem.has_equality:
        lamda_inequality = qsp.constraints[0].dual_value

    elif not problem.has_inequality and problem.has_equality:
        lamda_equality = -qsp.constraints[equality_index].dual_value

    return delta.value, lamda_equality, lamda_inequality


def convergence_test(
    result: Result,
    delta_j: np.ndarray,
    lamda_equality_i: np.ndarray,
    lamda_inequality_i: np.ndarray,
    epsilon: float,
) -> bool:
    abs_df_dot_delta = abs(np.dot(result.df, delta_j))
    abs_equality__err = abs(
        np.sum([lamda * c for lamda, c in zip(lamda_equality_i, result.eq)])
    )
    abs_inequality__err = abs(
        np.sum([lamda * c for lamda, c in zip(lamda_inequality_i, result.ie)])
    )

    return abs_df_dot_delta + abs_equality__err + abs_inequality__err < epsilon


def _calculate_mu_i(mu_im1: Union[np.ndarray, None], lamda: np.ndarray):
    if mu_im1 is None:
        return np.abs(lamda)

    # element-wise maximum is assumed
    return np.maximum(np.abs(lamda), 0.5 * (mu_im1 + np.abs(lamda)))


def perform_linesearch(
    problem: AbstractProblem,
    result: Result,
    mu_equality: Union[np.ndarray, None],
    mu_inequality: Union[np.ndarray, None],
    lamda_equality: np.ndarray,
    lamda_inequality: np.ndarray,
    delta: np.ndarray,
    x_jm1: np.ndarray,
):
    mu_equality = _calculate_mu_i(mu_equality, lamda_equality)
    mu_inequality = _calculate_mu_i(mu_inequality, lamda_inequality)

    # TODO: Cache this function to avoid repeated calls to the objective (and constraints)
    def phi(alpha: np.floating):
        x = x_jm1 + alpha * delta
        new_result = problem(x)
        sum_equality = (mu_equality * np.abs(new_result.eq)).sum()
        sum_inequality = (
            mu_inequality * np.abs(np.array([min(0, c) for c in new_result.ie]))
        ).sum()

        return new_result.f + sum_equality + sum_inequality

    # dphi(0) for unconstrained minimisation = F'(x_jm1)*delta
    # this is extended to constrained minimisation subtracting the
    # weighted constraints at 0.
    capital_delta = (result.df * delta).sum() - phi(0) + result.f

    alpha = 1.0
    for _ in range(10):
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
            "Line search did not converge on an approimate minima",
            x=x_jm1,
            result=result,
            lamda_equality=lamda_equality,
            lamda_inequality=lamda_inequality,
        )

    return alpha, mu_equality, mu_inequality


def _derivative_lagrangian(
    result: Result,
    lamda_equality: np.ndarray,
    lamda_inequality: np.ndarray,
):
    c_equality_prime = sum(
        [lamda * dc for lamda, dc in zip(lamda_equality, result.deq)]
    )
    c_inequality_prime = sum(
        [lamda * dc for lamda, dc in zip(lamda_inequality, result.die)]
    )

    return result.df - c_equality_prime - c_inequality_prime


def _powells_gamma(gamma: np.ndarray, ksi: np.ndarray, B: np.ndarray):
    ksiTBksi = ksi.T @ B @ ksi  # used throughout eqn 10
    ksiTgamma = ksi.T @ gamma  # dito, to reduce amount of matmul

    theta = 1.0
    if ksiTgamma < 0.2 * ksiTBksi:
        theta = 0.8 * ksiTBksi / (ksiTBksi - ksiTgamma)

    return theta * gamma + (1 - theta) * (B @ ksi)  # eqn 9


def calculate_new_B(
    problem: AbstractProblem,
    result: Result,
    B: np.ndarray,
    x_jm1: np.ndarray,
    x_j: np.ndarray,
    lamda_equality: np.ndarray,
    lamda_inequality: np.ndarray,
):
    new_result = problem(x_j)
    # xi (the symbol name) would be a bit confusing in this context,
    # ksi is how its pronounced in modern greek
    # reshape ksi to be a matrix
    ksi = (x_j - x_jm1).reshape((x_j.shape[0], 1))

    g1 = _derivative_lagrangian(
        new_result,
        lamda_equality,
        lamda_inequality,
    )
    g2 = _derivative_lagrangian(
        result,
        lamda_equality,
        lamda_inequality,
    )
    gamma = (g1 - g2).reshape((x_j.shape[0], 1))

    gamma = _powells_gamma(gamma, ksi, B)

    if (gamma == 0).all():
        gamma[:] = 1e-10

    if (ksi == 0).all():
        ksi[:] = 1e-10

    B = (
        B
        - ((B @ ksi @ ksi.T @ B) / (ksi.T @ B @ ksi))
        + ((gamma @ gamma.T) / (gamma.T @ ksi))
    )  # eqn 8

    return B
