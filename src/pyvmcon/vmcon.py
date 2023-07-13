from typing import Union, Optional
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


def solve(
    problem: AbstractProblem,
    x: np.ndarray,
    lbs: Optional[np.ndarray] = None,
    ubs: Optional[np.ndarray] = None,
    *,
    max_iter: int = 10,
    epsilon: float = 1e-8,
    qsp_tolerence: float = 1e-4,
    initial_B: Optional[np.ndarray] = None,
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

    Returns
    -------
    x : ndarray
        The solution vector which VMCON converges to.

    lamda_equality : ndarray
        The Lagrange multipliers for the equality constraints at the solution vector.

    lamda_inequality : ndarray
        The Lagrange multipliers for the inequality constraints at the solution vector.

    result : Result
        The result from running the solution vector through the problem.
    """

    if len(x.shape) != 1:
        raise ValueError("Input vector `x` is not a 1D array")

    if lbs is not None and (x < lbs).any():
        msg = "x is initially in an infeasible region because at least one x is lower than a lower bound"  # noqa: E501
        logger.error(
            f"{msg}. The out of bounds variables are at indices {', '.join(_find_out_of_bounds_vars(x, lbs))} (0-based indexing)"  # noqa: E501
        )
        raise ValueError(msg)

    if ubs is not None and (x > ubs).any():
        msg = "x is initially in an infeasible region because at least one x is greater than an upper bound"  # noqa: E501
        logger.error(
            f"{msg}. The out of bounds variables are at indices {', '.join(_find_out_of_bounds_vars(ubs, x))} (0-based indexing)"  # noqa: E501
        )
        raise ValueError(msg)

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
        result = problem(x)

        # solve the quadratic subproblem to identify
        # our search direction and the Lagrange multipliers
        # for our constraints
        try:
            delta, lamda_equality, lamda_inequality = solve_qsp(
                problem, result, x, B, lbs, ubs, qsp_tolerence
            )
        except _QspSolveException as e:
            raise QSPSolverException(
                "QSP failed to solve, indicating no feasible solution could be found.",
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
        alpha, mu_equality, mu_inequality, new_result = perform_linesearch(
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
            result,
            new_result,
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
    lbs: Optional[np.ndarray],
    ubs: Optional[np.ndarray],
    tolerance: float,
):
    """Solves the quadratic programming problem detailed in equation 4 and 5
    of the VMCON paper.

    The QSP is solved using cvxpy. cvxpy requires the problem be convex, which is
    ensured by equation 9 of the VMCON paper.

    Parameters
    ----------
    problem : AbstractProblem
        The current minimisation problem being solved.

    result : Result
        Contains the data for the (j-1)th evaluation point.

    x : ndarray
        The (j-1)th evaluation point.

    B : ndarray
        The current approximation of the Hessian matrix.

    lbs : ndarray
        The lower bounds of `x`.

    ubs : ndarray
        The upper bounds of `x`.

    tolerance : float
        The relative tolerance of the QSP solver.
        See https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
        `eps_rel`.
    """
    delta = cp.Variable(x.shape)
    problem_statement = cp.Minimize(
        result.f
        + (0.5 * cp.quad_form(delta, B, assume_PSD=True))
        + (delta.T @ result.df)
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
    qsp.solve(verbose=False, solver=cp.OSQP, eps_rel=tolerance)

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
    lamda_equality: np.ndarray,
    lamda_inequality: np.ndarray,
    epsilon: float,
) -> bool:
    """Test if the convergence criteria of VMCON have been met.
    Equation 11 of the VMCON paper. Note this tests convergence at the
    point (j-1)th evaluation point.

    Parameters
    ----------
    result : Result
        Contains the data for the (j-1)th evaluation point.

    delta_j : ndarray
        The search direction for the jth evaluation point.

    lambda_equality : ndarray
        The Lagrange multipliers for equality constraints for the jth
        evaluation point.

    lambda_inequality : ndarray
        The Lagrange multipliers for inequality constraints for the jth
        evaluation point.

    epsilon : float
        The user-supplied error tolerance.
    """
    abs_df_dot_delta = abs(np.dot(result.df, delta_j))
    abs_equality_err = np.sum(
        [abs(lamda * c) for lamda, c in zip(lamda_equality, result.eq)]
    )
    abs_inequality_err = np.sum(
        [abs(lamda * c) for lamda, c in zip(lamda_inequality, result.ie)]
    )

    return (abs_df_dot_delta + abs_equality_err + abs_inequality_err) < epsilon


def _calculate_mu_i(mu_im1: Union[np.ndarray, None], lamda: np.ndarray):
    if mu_im1 is None:
        return np.abs(lamda)

    # element-wise maximum is assumed
    return np.maximum(np.abs(lamda), 0.5 * (mu_im1 + np.abs(lamda)))


def perform_linesearch(
    problem: AbstractProblem,
    result: Result,
    mu_equality: Optional[np.ndarray],
    mu_inequality: Optional[np.ndarray],
    lamda_equality: np.ndarray,
    lamda_inequality: np.ndarray,
    delta: np.ndarray,
    x_jm1: np.ndarray,
):
    """Performs the line search on equation 6 (to minimise phi).

    Parameters
    ----------
    problem : AbstractProblem
        The current minimisation problem being solved.

    result : Result
        Contains the data for the (j-1)th evaluation point.

    mu_equality : ndarray
        The mu values for the equality constraints.

    mu_inequality : ndarray
        The mu values for the inequality constraints.
    """
    mu_equality = _calculate_mu_i(mu_equality, lamda_equality)
    mu_inequality = _calculate_mu_i(mu_inequality, lamda_inequality)

    def phi(result: Result):
        sum_equality = (mu_equality * np.abs(result.eq)).sum()
        sum_inequality = (
            mu_inequality * np.abs(np.array([min(0, c) for c in result.ie]))
        ).sum()

        return result.f + sum_equality + sum_inequality

    phi_0 = phi(result)

    result_at_1 = problem(x_jm1 + delta)

    # powell suggests this Delta due to possible
    # discontinuity in the derivative at alpha=0
    # Powell 1978
    capital_delta = phi(result_at_1) - phi_0

    alpha = 1.0
    for _ in range(10):
        # exit if we satisfy the Armijo condition
        # or the Kovari condition
        new_result = problem(x_jm1 + alpha * delta) if alpha != 1.0 else result_at_1
        phi_alpha = phi(new_result)
        if phi_alpha <= phi_0 + 0.1 * alpha * capital_delta or phi_alpha > phi_0:
            break

        alpha = min(
            0.1 * alpha,
            -(alpha**2) / (2 * (phi_alpha - phi_0 - capital_delta * alpha)),
        )

    else:
        raise LineSearchConvergenceException(
            "Line search did not converge on an approimate minima",
            x=x_jm1,
            result=result,
            lamda_equality=lamda_equality,
            lamda_inequality=lamda_inequality,
        )

    return alpha, mu_equality, mu_inequality, new_result


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
    result: Result,
    new_result: Result,
    B: np.ndarray,
    x_jm1: np.ndarray,
    x_j: np.ndarray,
    lamda_equality: np.ndarray,
    lamda_inequality: np.ndarray,
):
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
        logger.warning("All gamma components are 0")
        gamma[:] = 1e-10

    if (ksi == 0).all():
        logger.warning("All xi (ksi) components are 0")
        ksi[:] = 1e-10

    B = (
        B
        - ((B @ ksi @ ksi.T @ B) / (ksi.T @ B @ ksi))
        + ((gamma @ gamma.T) / (ksi.T @ gamma))
    )  # eqn 8

    return B


def _find_out_of_bounds_vars(higher: np.ndarray, lower: np.ndarray):
    """Return the indices of the out of bounds variables"""

    out_of_bounds = []
    for i, boolean in enumerate((higher - lower) < 0):
        if boolean:
            out_of_bounds.append(str(i))

    return out_of_bounds
