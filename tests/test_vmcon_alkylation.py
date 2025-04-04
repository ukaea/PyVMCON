import cvxpy
import numpy as np
import pytest
from pyvmcon import solve
from pyvmcon.problem import Problem

"""
This is a complex constrained optimisation problem from Colville
Found in chapter 4 here: https://apps.dtic.mil/sti/tr/pdf/AD0679037.pdf
Note that there is a typo in one of the equality constraints.
See https://courses.mai.liu.se/GU/TAOP04/process-optimization.pdf

I have added this because my version of VMCON was not able to solve this
and I wanted to check if this one could. It doesn't appear to be able to
either. UPDATE: Ok, so in fact it is OSQP that struggles with this problem.
Switching the solver provides the correct solution. Not all work, and I have
not tried them all.

For info, I provide some quick (optional) tests for the same 
problem using SLSQP and COBYLA (which in scipy is not able to handle 
equality constraints or bounds - facepalm).
"""


class AlkylationData:
    c1 = 0.063
    c2 = 5.04
    c3 = 0.035
    c4 = 10.0
    c5 = 3.36
    d4l = 99.0 / 100.0
    d4u = 100.0 / 99.0
    d7l = 99.0 / 100.0
    d7u = 100.0 / 99.0
    d9l = 9.0 / 10.0
    d9u = 10.0 / 9.0
    d10l = 99.0 / 100.0
    d10u = 100.0 / 99.0
    dimension = 10
    lower_bounds = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 85.0, 90.0, 3.0, 1.2, 145.0])
    upper_bounds = np.array(
        [2000.0, 16000.0, 120.0, 5000.0, 2000.0, 93.0, 95.0, 12.0, 4.0, 162.0]
    )
    suggested_x0 = np.array(
        [
            1745.0,
            12000.0,
            110.0,
            3048.0,
            1974.0,
            89.2,
            92.8,
            8.0,
            3.6,
            145.0,
        ]
    )
    # Given to 1DP
    true_x = np.array(
        [
            1698.0,
            15818.0,
            54.1,
            3031.0,
            2000.0,
            90.1,
            95.0,
            10.5,
            1.6,
            154.0,
        ]
    )
    # Given to 0DP
    true_f_x = 1769.0


data = AlkylationData()


def f_objective(x):
    """
    This is a maximisation objective, so we flip the sign.
    """
    return (
        -data.c1 * x[3] * x[6]
        + data.c2 * x[0]
        + data.c3 * x[1]
        + data.c4 * x[2]
        + data.c5 * x[4]
    )


def df_objective(x):
    grad = np.zeros(data.dimension)
    grad[0] = data.c2
    grad[1] = data.c3
    grad[2] = data.c4
    grad[3] = -data.c1 * x[6]
    grad[4] = data.c5
    grad[6] = -data.c1 * x[3]
    return grad


def f_equality1(x):
    return 1.22 * x[3] - x[0] - x[4]


def df_equality1(x):
    grad = np.zeros(data.dimension)
    grad[0] = -1.0
    grad[3] = 1.22
    grad[4] = -1.0
    return grad


def f_equality2(x):
    # There is a typo where I found this orginally...
    return 98_000.0 * x[2] / (x[3] * x[8] + 1000.0 * x[2]) - x[5]


def df_equality2(x):
    grad = np.zeros(data.dimension)
    grad[2] = (98_000.0 * x[3] * x[8]) / (x[3] * x[8] + 1000.0 * x[2]) ** 2
    grad[3] = -98_000.0 * x[2] * x[8] / (x[8] * x[3] + 1000.0 * x[2]) ** 2
    grad[5] = -1.0
    grad[8] = -98_000.0 * x[2] * x[3] / (x[8] * x[3] + 1000.0 * x[2]) ** 2
    return grad


def f_equality3(x):
    return (x[1] + x[4]) / x[0] - x[7]


def df_equality3(x):
    grad = np.zeros(data.dimension)
    grad[0] = -(x[1] + x[4]) / x[0] ** 2
    grad[1] = 1.0 / x[0]
    grad[4] = 1.0 / x[0]
    grad[7] = -1.0
    return grad


def f_inequality1(x):
    a = 1.12 + 0.13167 * x[7] - 0.00667 * x[7] ** 2
    return x[0] * a - data.d4l * x[3]


def df_inequality1(x):
    grad = np.zeros(data.dimension)
    a = 1.12 + 0.13167 * x[7] - 0.00667 * x[7] ** 2
    grad[0] = a
    grad[3] = -data.d4l
    grad[7] = x[0] * (0.13167 - 2.0 * 0.00667 * x[7])
    return grad


def f_inequality2(x):
    a = 1.12 + 0.13167 * x[7] - 0.00667 * x[7] ** 2
    return -x[0] * a + data.d4u * x[3]


def df_inequality2(x):
    grad = np.zeros(data.dimension)
    a = 1.12 + 0.13167 * x[7] - 0.00667 * x[7] ** 2
    grad[0] = -a
    grad[3] = data.d4u
    grad[7] = -x[0] * (0.13167 - 2.0 * 0.00667 * x[7])
    return grad


def f_inequality3(x):
    return (
        86.35 + 1.098 * x[7] - 0.038 * x[7] ** 2 + 0.325 * (x[5] - 89.0)
    ) - data.d7l * x[6]


def df_inequality3(x):
    grad = np.zeros(data.dimension)
    grad[5] = 0.325
    grad[6] = -data.d7l
    grad[7] = 1.098 - 2.0 * 0.038 * x[7]
    return grad


def f_inequality4(x):
    return (
        -(86.35 + 1.098 * x[7] - 0.038 * x[7] ** 2 + 0.325 * (x[5] - 89.0))
        + data.d7u * x[6]
    )


def df_inequality4(x):
    grad = np.zeros(data.dimension)
    grad[5] = -0.325
    grad[6] = data.d7u
    grad[7] = -(1.098 - 2.0 * 0.038 * x[7])
    return grad


def f_inequality5(x):
    return (35.82 - 0.222 * x[9]) - data.d9l * x[8]


def df_inequality5(x):
    grad = np.zeros(data.dimension)
    grad[8] = -data.d9l
    grad[9] = -0.222
    return grad


def f_inequality6(x):
    return -(35.82 - 0.222 * x[9]) + data.d9u * x[8]


def df_inequality6(x):
    grad = np.zeros(data.dimension)
    grad[8] = data.d9u
    grad[9] = 0.222
    return grad


def f_inequality7(x):
    return (-133.0 + 3.0 * x[6]) - data.d10l * x[9]


def df_inequality7(x):
    grad = np.zeros(data.dimension)
    grad[6] = 3.0
    grad[9] = -data.d10l
    return grad


def f_inequality8(x):
    return -(-133.0 + 3.0 * x[6]) + data.d10u * x[9]


def df_inequality8(x):
    grad = np.zeros(data.dimension)
    grad[6] = -3.0
    grad[9] = data.d10u
    return grad


EQUALITY_CONSTRAINTS = [f_equality1, f_equality2, f_equality3]
D_EQUALITY_CONSTRAINTS = [df_equality1, df_equality2, df_equality3]
INEQUALITY_CONSTRAINTS = [
    f_inequality1,
    f_inequality2,
    f_inequality3,
    f_inequality4,
    f_inequality5,
    f_inequality6,
    f_inequality7,
    f_inequality8,
]
D_INEQUALITY_CONSTRAINTS = [
    df_inequality1,
    df_inequality2,
    df_inequality3,
    df_inequality4,
    df_inequality5,
    df_inequality6,
    df_inequality7,
    df_inequality8,
]


def test_gradients():
    np.random.seed(6245431)
    x = np.random.rand(data.dimension)

    from scipy.optimize import approx_fprime

    grad = approx_fprime(x, f_objective, 1e-8)
    grad2 = df_objective(x)
    np.testing.assert_allclose(grad, grad2, rtol=1e-4)
    for f, df in zip(EQUALITY_CONSTRAINTS, D_EQUALITY_CONSTRAINTS):
        grad = approx_fprime(x, f, 1e-8)
        grad2 = df(x)
        np.testing.assert_allclose(grad, grad2, rtol=1e-4)
    for f, df in zip(INEQUALITY_CONSTRAINTS, D_INEQUALITY_CONSTRAINTS):
        grad = approx_fprime(x, f, 1e-8)
        grad2 = df(x)
        np.testing.assert_allclose(grad, grad2, rtol=1e-4)


@pytest.mark.parametrize(
    "qsp_solver,expected_success",
    [("OSQP", False), ("CLARABEL", True), ("ECOS_BB", True)],
)
def test_vmcon_alkylation_problem(qsp_solver, expected_success):
    problem = Problem(
        f=f_objective,
        df=df_objective,
        equality_constraints=EQUALITY_CONSTRAINTS,
        dequality_constraints=D_EQUALITY_CONSTRAINTS,
        inequality_constraints=INEQUALITY_CONSTRAINTS,
        dinequality_constraints=D_INEQUALITY_CONSTRAINTS,
    )

    if expected_success:
        (x, _, _, result) = solve(
            problem,
            data.suggested_x0,
            lbs=data.lower_bounds,
            ubs=data.upper_bounds,
            max_iter=4000,
            epsilon=1e-8,
            # initial_B=np.zeros((10, 10)),  # This does not appear to help
            qsp_options={"solver": qsp_solver},
        )
        assert np.round(abs(result.f)) == data.true_f_x
        np.testing.assert_allclose(
            np.round(x, decimals=1),
            data.true_x,
            atol=0.0,
            rtol=0.0033,
        )
    else:
        with pytest.raises(cvxpy.error.SolverError):
            (x, _, _, result) = solve(
                problem,
                data.suggested_x0,
                lbs=data.lower_bounds,
                ubs=data.upper_bounds,
                max_iter=4000,
                epsilon=1e-8,
                # initial_B=np.zeros((10, 10)),
                qsp_options={"solver": qsp_solver},
            )


def test_slsqp_alkylation_problem():
    from scipy.optimize import minimize

    result = minimize(
        f_objective,
        data.suggested_x0,
        method="SLSQP",
        jac=df_objective,
        bounds=[(a, b) for (a, b) in zip(data.lower_bounds, data.upper_bounds)],
        options={"maxiter": 1000},
        constraints=[
            {"type": "eq", "fun": f_equality1, "jac": df_equality1},
            {"type": "eq", "fun": f_equality2, "jac": df_equality2},
            {"type": "eq", "fun": f_equality3, "jac": df_equality3},
            {"type": "ineq", "fun": f_inequality1, "jac": df_inequality1},
            {"type": "ineq", "fun": f_inequality2, "jac": df_inequality2},
            {"type": "ineq", "fun": f_inequality3, "jac": df_inequality3},
            {"type": "ineq", "fun": f_inequality4, "jac": df_inequality4},
            {"type": "ineq", "fun": f_inequality5, "jac": df_inequality5},
            {"type": "ineq", "fun": f_inequality6, "jac": df_inequality6},
            {"type": "ineq", "fun": f_inequality7, "jac": df_inequality7},
            {"type": "ineq", "fun": f_inequality8, "jac": df_inequality8},
        ],
    )
    print(result)
    assert result.success


def test_cobyla_alkylation_problem():
    from scipy.optimize import minimize

    delta_x = 1e-6  # This is my tolerance for converting equalities to inequalities
    # The lower it is, the more accurate the solution. I'm guessing this is what was
    # taken in the reference. Lowering it further makes the results worse.

    result = minimize(
        f_objective,
        data.suggested_x0,
        method="Cobyla",
        options={"maxiter": 20000},
        tol=1e-20,
        constraints=[
            # Jesus Christ scipy, this is not so hard to implement...
            # Convert bounds to inequalities
            {"type": "ineq", "fun": lambda x: x},
            {"type": "ineq", "fun": lambda x: 1 - x},
            # Convert equalities to inequalities
            {"type": "ineq", "fun": lambda x: f_equality1(x) - delta_x},
            {"type": "ineq", "fun": lambda x: -f_equality1(x) - delta_x},
            {"type": "ineq", "fun": lambda x: f_equality2(x) - delta_x},
            {"type": "ineq", "fun": lambda x: -f_equality2(x) - delta_x},
            {"type": "ineq", "fun": lambda x: f_equality3(x) - delta_x},
            {"type": "ineq", "fun": lambda x: -f_equality3(x) - delta_x},
            {"type": "ineq", "fun": f_inequality1},
            {"type": "ineq", "fun": f_inequality2},
            {"type": "ineq", "fun": f_inequality3},
            {"type": "ineq", "fun": f_inequality4},
            {"type": "ineq", "fun": f_inequality5},
            {"type": "ineq", "fun": f_inequality6},
            {"type": "ineq", "fun": f_inequality7},
            {"type": "ineq", "fun": f_inequality8},
        ],
    )
    assert result.success
    assert np.round(abs(result.fun)) == data.true_f_x
    np.testing.assert_allclose(
        np.round(result.x, decimals=1),
        data.true_x,
        atol=0.0,
        rtol=0.0033,
    )
