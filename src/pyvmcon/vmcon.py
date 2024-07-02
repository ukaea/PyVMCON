import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cvxpy as cp
import numpy as np

from .exceptions import (
    LineSearchConvergenceException,
    QSPSolverException,
    VMCONConvergenceException,
    _QspSolveException,
)
from .problem import AbstractProblem, Result, T

logger = logging.getLogger(__name__)


def solve(
    problem: AbstractProblem,
    x: np.ndarray,
    lbs: Optional[np.ndarray] = None,
    ubs: Optional[np.ndarray] = None,
    *,
    max_iter: int = 10,
    epsilon: float = 1e-8,
    qsp_options: Optional[Dict[str, Any]] = None,
    initial_B: Optional[np.ndarray] = None,
    callback: Optional[Callable[[int, Result, np.ndarray, float], None]] = None,
    additional_convergence: Optional[
        Callable[[Result, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]
    ] = None,
    overwrite_convergence_criteria: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Result]:
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

    qsp_options : Optional[Dict[str, Any]]
        Dictionary of keyword arguments that are passed to the
        CVXPY `Probelem.solve` method. `None` will pass no
        additional arguments to the solver.

    initial_B : ndarray
        Initial estimate of the Hessian matrix `B`. If `None`, `B` is the
        identity matrix of shape `(n, n)`.

    callback : Optional[Callable[[int, ndarray, Result], None]]
        A callable which takes the current iteration, the `Result` of the
        current design point, current design point, and the convergence parameter
        as arguments and returns `None`. This callable is called each iteration
        after the QSP is solved but before the convergence test.

    additional_convergence : Optional[Callable[[Result, ndarray, ndarray, ndarray, ndarray], None]]
        A callabale which takes: the `Result` of the current design point,
        the current design point, the proposed search direction for the
        next design point, the equality Lagrange multipliers, and the
        inequality Lagrange multipliers. The callable returns a boolean
        indicating whether VMCON should be allowed to converge. Note that
        the original VMCON convergence criteria being `False` will stop
        convergence even if this callable returns `True` unless we
        `overwrite_convergence_criteria`.

    overwrite_convergence_criteria : bool
        Ignore original VMCON convergence criteria and only
        evaluate convergence using `additional_convergence`.

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

    if overwrite_convergence_criteria and additional_convergence is None:
        raise ValueError(
            "Cannot overwrite convergence criteria without "
            "providing an 'additional_convergence' callable."
        )

    # n is denoted in the VMCON paper
    # as the number of inputs the function
    # and the constraints take
    n = x.shape[0]

    # The paper uses the B matrix as the
    # running approximation of the Hessian
    B = np.identity(n) if initial_B is None else initial_B

    callback = callback or (lambda _i, _result, _x, _con: None)
    additional_convergence = additional_convergence or (
        lambda _result, _x, _delta, _lambda_eq, _lambda_in: True
    )

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
                problem, result, x, B, lbs, ubs, qsp_options or {}
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
        convergence_info = convergence_value(
            result, delta, lamda_equality, lamda_inequality
        )

        callback(i, result, x, convergence_info)

        if additional_convergence(
            result, x, delta, lamda_equality, lamda_inequality
        ) and (overwrite_convergence_criteria or convergence_info < epsilon):
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
        # so our running x is not overridden yet!
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
    options: Dict[str, Any],
) -> Tuple[np.ndarray, ...]:
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

    options : Dict[str, Any]
        Dictionary of keyword arguments that are passed to the
        CVXPY `Problem.solve` method.

    Notes
    -----
    * By default, OSQP (https://osqp.org/) is the `solver` used in
    the `solve` method however this can be changed by specifying a
    different `solver` in the `options` dictionary.
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
    qsp.solve(**{"solver": cp.OSQP, **options})

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


def convergence_value(
    result: Result,
    delta_j: np.ndarray,
    lamda_equality: np.ndarray,
    lamda_inequality: np.ndarray,
) -> float:
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
    """
    ind_eq = min(lamda_equality.shape[0], result.eq.shape[0])
    ind_ieq = min(lamda_inequality.shape[0], result.ie.shape[0])
    abs_df_dot_delta = abs(np.dot(result.df, delta_j))
    abs_equality_err = abs(np.sum(lamda_equality[:ind_eq] * result.eq[:ind_eq]))
    abs_inequality_err = abs(np.sum(lamda_inequality[:ind_ieq] * result.ie[:ind_ieq]))

    return abs_df_dot_delta + abs_equality_err + abs_inequality_err


def _calculate_mu_i(mu_im1: Union[np.ndarray, None], lamda: np.ndarray) -> np.ndarray:
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
) -> Tuple[float, np.ndarray, np.ndarray, Result]:
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

    def phi(result: Result) -> T:
        sum_equality = (mu_equality * np.abs(result.eq)).sum()
        sum_inequality = (mu_inequality * np.abs(np.minimum(0, result.ie))).sum()

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
            "Line search did not converge on an approximate minima",
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
) -> np.ndarray:
    ind_eq = min(lamda_equality.shape[0], result.deq.shape[0])
    ind_ieq = min(lamda_inequality.shape[0], result.die.shape[0])
    c_equality_prime = (lamda_equality[:ind_eq, None] * result.deq[:ind_eq]).sum(
        axis=None if ind_eq == 0 else 0
    )
    c_inequality_prime = (lamda_inequality[:ind_ieq, None] * result.die[:ind_ieq]).sum(
        axis=None if ind_ieq == 0 else 0
    )

    return result.df - c_equality_prime - c_inequality_prime


def _powells_gamma(gamma: np.ndarray, ksi: np.ndarray, B: np.ndarray) -> np.ndarray:
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
) -> np.ndarray:
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
    gamma = _powells_gamma((g1 - g2).reshape((x_j.shape[0], 1)), ksi, B)

    if (gamma == 0).all():
        logger.warning("All gamma components are 0")
        gamma[:] = 1e-10

    if (ksi == 0).all():
        logger.warning("All xi (ksi) components are 0")
        ksi[:] = 1e-10

    # eqn 8
    B_ksi = B @ ksi
    B += (gamma @ gamma.T) / (ksi.T @ gamma) - ((B_ksi @ ksi.T @ B) / (ksi.T @ B_ksi))

    return B


def _find_out_of_bounds_vars(higher: np.ndarray, lower: np.ndarray) -> List[str]:
    """Return the indices of the out of bounds variables"""
    return np.nonzero((higher - lower) < 0)[0].astype(str).tolist()
