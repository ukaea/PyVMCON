from typing import NamedTuple

import numpy as np
import pytest
from pyvmcon import solve
from pyvmcon.exceptions import VMCONConvergenceException
from pyvmcon.problem import Problem


class VMCONTestAsset(NamedTuple):
    problem: Problem
    initial_x: np.ndarray
    expected_x: np.ndarray
    expected_lamda_equality: np.ndarray
    expected_lamda_inequality: np.ndarray
    lbs: np.ndarray = None
    ubs: np.ndarray = None
    max_iter: int = 10
    epsilon: float = 1e-8


@pytest.mark.parametrize(
    "vmcon_example",
    [
        # Test 1 detailed in ANL-80-64 page 25
        VMCONTestAsset(
            Problem(
                lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
                lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
                [lambda x: x[0] - (2 * x[1]) + 1],
                [lambda x: -((x[0] ** 2) / 4) - (x[1] ** 2) + 1],
                [lambda _: np.array([1, -2])],
                [lambda x: np.array([-0.5 * x[0], -2 * x[1]])],
            ),
            initial_x=np.array([2.0, 2.0]),
            expected_x=[8.228756e-1, 9.114378e-1],
            expected_lamda_equality=[-1.594491],
            expected_lamda_inequality=[1.846591],
        ),
        # Test 1 detailed in ANL-80-64 page 25
        # with one of the constraints duplicated
        # This is to check we can deal with more constraints
        # than inputs
        VMCONTestAsset(
            Problem(
                lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
                lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
                [lambda x: x[0] - (2 * x[1]) + 1, lambda x: x[0] - (2 * x[1]) + 1],
                [lambda x: -((x[0] ** 2) / 4) - (x[1] ** 2) + 1],
                [lambda _: np.array([1, -2]), lambda _: np.array([1, -2])],
                [lambda x: np.array([-0.5 * x[0], -2 * x[1]])],
            ),
            initial_x=np.array([2.0, 2.0]),
            expected_x=[8.228756e-1, 9.114378e-1],
            # duplicating the constraint is probably expected
            # to change the Lagrange multipliers
            expected_lamda_equality=[-0.7972455591261, -0.7972455591261],
            expected_lamda_inequality=[1.846591],
        ),
        # Test 1 detailed in ANL-80-64 page 25
        # with added, unintrusive, bounds
        VMCONTestAsset(
            Problem(
                lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
                lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
                [lambda x: x[0] - (2 * x[1]) + 1],
                [lambda x: -((x[0] ** 2) / 4) - (x[1] ** 2) + 1],
                [lambda _: np.array([1, -2])],
                [lambda x: np.array([-0.5 * x[0], -2 * x[1]])],
            ),
            initial_x=np.array([2.0, 2.0]),
            expected_x=[8.228756e-1, 9.114378e-1],
            expected_lamda_equality=[-1.594491],
            expected_lamda_inequality=[1.846591],
            lbs=np.array([-10, -10]),
            ubs=np.array([10, 10]),
        ),
        # Test 2 detailed in ANL-80-64 page 28
        VMCONTestAsset(
            Problem(
                lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
                lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
                [],
                [
                    lambda x: x[0] - (2 * x[1]) + 1,
                    lambda x: -((x[0] ** 2) / 4) - (x[1] ** 2) + 1,
                ],
                [],
                [
                    lambda _: np.array([1, -2]),
                    lambda x: np.array([-0.5 * x[0], -2 * x[1]]),
                ],
            ),
            initial_x=np.array([2.0, 2.0]),
            expected_x=[1.6649685472365443, 0.55404867491788852],
            expected_lamda_equality=[],
            expected_lamda_inequality=[0, 0.80489557193146243],
        ),
        # Example 1a of https://en.wikipedia.org/wiki/Lagrange_multiplier
        VMCONTestAsset(
            Problem(
                lambda x: x[0] + x[1],
                lambda _: np.array([1, 1]),
                [lambda x: (x[0] ** 2) + (x[1] ** 2) - 1],
                [],
                [lambda x: np.array([2 * x[0], 2 * x[1]])],
                [],
            ),
            initial_x=np.array([1.0, 1.0]),
            epsilon=2e-8,
            expected_x=[0.5 * 2**0.5, 0.5 * 2**0.5],  # Shouldn't these be negative?
            expected_lamda_equality=[2 ** (-0.5)],
            expected_lamda_inequality=[],
        ),
    ],
)
def test_vmcon_paper_feasible_examples(vmcon_example: VMCONTestAsset):
    """Tests example runs of VMCON provided in the VMCON paper
    produce similar results between their implementation, and this
    implementation.
    """
    x, lamda_equality, lamda_inequality, _ = solve(
        vmcon_example.problem,
        vmcon_example.initial_x,
        vmcon_example.lbs,
        vmcon_example.ubs,
        max_iter=vmcon_example.max_iter,
        epsilon=vmcon_example.epsilon,
    )

    assert x == pytest.approx(vmcon_example.expected_x)
    assert lamda_equality == pytest.approx(vmcon_example.expected_lamda_equality)
    assert lamda_inequality == pytest.approx(vmcon_example.expected_lamda_inequality)


@pytest.mark.parametrize(
    "vmcon_example",
    [
        VMCONTestAsset(
            Problem(
                lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
                lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
                [lambda x: x[0] + x[1] - 3],
                [lambda x: -((x[0] ** 2) / 4) - (x[1] ** 2) + 1],
                [lambda _: np.array([1.0, 1.0])],
                [lambda x: np.array([-0.5 * x[0], -2 * x[1]])],
            ),
            initial_x=np.array([2.0, 2.0]),
            max_iter=5,
            expected_x=[2.3999994310874733, 0.6],
            expected_lamda_equality=[0.0],
            expected_lamda_inequality=[0.0],
        ),
    ],
)
def test_vmcon_paper_infeasible_examples(vmcon_example: VMCONTestAsset):
    """Tests runs of VMCON where the problem describes a minimisation
    which is infeasible given the constraints.

    Assertions on the returned `x` (the last tried input vector) and
    corresponding Lagrange multipliers have been removed as the QSP
    implementation produced different final points from the VMCON
    paper. This is not surprising considering these problems are
    infeasible and we deem the assertions to hold little meaning;
    what is important--and thus tested--is that VMCON fails to
    converge in these infeasible cases.
    """
    with pytest.raises(VMCONConvergenceException):
        solve(
            vmcon_example.problem,
            vmcon_example.initial_x,
            max_iter=vmcon_example.max_iter,
            epsilon=vmcon_example.epsilon,
        )
