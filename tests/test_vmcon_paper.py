import pytest
from typing import Callable, List, NamedTuple
import numpy as np

from vmcon import VMCON


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
        VMCONTestAsset(
            lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
            equality_constraints=[lambda x: x[0] - (2 * x[1]) + 1],
            inequality_constraints=[
                lambda x: -((x[0] ** 2) / 4) - (x[1] ** 2) + 1,
            ],
            initial_x=np.array([2.0, 2.0]),
            max_iter=10,
            epsilon=1e-8,
            expected_x=[8.228756e-1, 9.114378e-1],
            expected_lamda_equality=[-1.594491],
            expected_lamda_inequality=[1.846591],
        )
    ],
)
def test_vmcon_paper_examples(vmcon_example: VMCONTestAsset):
    n = vmcon_example.initial_x.size

    vmcon = VMCON(
        vmcon_example.function,
        vmcon_example.equality_constraints,
        vmcon_example.inequality_constraints,
        n,
    )

    x, lamda_equality, lamda_inequality = vmcon.run_vmcon(
        vmcon_example.initial_x, vmcon_example.max_iter, vmcon_example.epsilon
    )

    assert x == pytest.approx(vmcon_example.expected_x)
    assert lamda_equality == pytest.approx(vmcon_example.expected_lamda_equality)
    assert lamda_inequality == pytest.approx(vmcon_example.expected_lamda_inequality)
