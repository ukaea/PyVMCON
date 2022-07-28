import pytest
from typing import Callable, List, NamedTuple
import numpy as np

from vmcon import solve
from vmcon.exceptions import VMCONConvergenceException
from vmcon.building_blocks import Function


class VMCONTestAsset(NamedTuple):
    function: Callable[[np.ndarray], np.floating]
    equality_constraints: List[Callable[[np.ndarray], np.floating]]
    inequality_constraints: List[Callable[[np.ndarray], np.floating]]

    initial_x: np.ndarray
    max_iter: int
    epsilon: float

    expected_x: np.ndarray
    expected_lamda_equality: np.ndarray
    expected_lamda_inequality: np.ndarray


@pytest.mark.parametrize(
    "vmcon_example",
    [
        # Test 1 detailed in ANL-80-64 page 25
        VMCONTestAsset(
            Function(lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2),
            equality_constraints=[Function(lambda x: x[0] - (2 * x[1]) + 1)],
            inequality_constraints=[
                Function(lambda x: -((x[0] ** 2) / 4) - (x[1] ** 2) + 1),
            ],
            initial_x=np.array([2.0, 2.0]),
            max_iter=10,
            epsilon=1e-8,
            expected_x=[8.228756e-1, 9.114378e-1],
            expected_lamda_equality=[-1.594491],
            expected_lamda_inequality=[1.846591],
        ),
        # Test 2 detailed in ANL-80-64 page 28
        VMCONTestAsset(
            Function(lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2),
            equality_constraints=[],
            inequality_constraints=[
                Function(lambda x: -((x[0] ** 2) / 4) - (x[1] ** 2) + 1),
                Function(lambda x: x[0] - (2 * x[1]) + 1),
            ],
            initial_x=np.array([2.0, 2.0]),
            max_iter=10,
            epsilon=1e-8,
            expected_x=[1.6649685472365443, 0.55404867491788852],
            expected_lamda_equality=[],
            expected_lamda_inequality=[0.80489557193146243, 0],
        ),
        # Example 1a of https://en.wikipedia.org/wiki/Lagrange_multiplier
        VMCONTestAsset(
            Function(lambda x: x[0] + x[1]),
            equality_constraints=[Function(lambda x: (x[0] ** 2) + (x[1] ** 2) - 1)],
            inequality_constraints=[],
            initial_x=np.array([1.0, 1.0]),
            max_iter=10,
            epsilon=2e-8,
            expected_x=[0.5 * 2**0.5, 0.5 * 2**0.5],  # Shouldn't these be negative?
            expected_lamda_equality=[2 ** (-0.5)],
            expected_lamda_inequality=[],
        ),
    ],
)
def test_vmcon_paper_feasible_examples(vmcon_example: VMCONTestAsset):
    x, lamda_equality, lamda_inequality = solve(
        vmcon_example.function,
        vmcon_example.equality_constraints,
        vmcon_example.inequality_constraints,
        vmcon_example.initial_x,
        vmcon_example.max_iter,
        vmcon_example.epsilon,
    )

    assert x == pytest.approx(vmcon_example.expected_x)
    assert lamda_equality == pytest.approx(vmcon_example.expected_lamda_equality)
    assert lamda_inequality == pytest.approx(vmcon_example.expected_lamda_inequality)


@pytest.mark.parametrize(
    "vmcon_example",
    [
        VMCONTestAsset(
            Function(lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2),
            equality_constraints=[Function(lambda x: x[0] + x[1] - 3)],
            inequality_constraints=[
                Function(lambda x: -((x[0] ** 2) / 4) - (x[1] ** 2) + 1),
            ],
            initial_x=np.array([2.0, 2.0]),
            max_iter=5,
            epsilon=1e-8,
            expected_x=[2.3999994310874733, 1.1249516708907119e6],
            expected_lamda_equality=[0.0],
            expected_lamda_inequality=[0.0],
        ),
    ],
)
def test_vmcon_paper_infeasible_examples(vmcon_example: VMCONTestAsset):
    with pytest.raises(VMCONConvergenceException):
        solve(
            vmcon_example.function,
            vmcon_example.equality_constraints,
            vmcon_example.inequality_constraints,
            vmcon_example.initial_x,
            vmcon_example.max_iter,
            vmcon_example.epsilon,
        )
