"""Test VMCON's use of bounds in simple linear functions."""

import numpy as np
import pytest

from pyvmcon.problem import Problem
from pyvmcon.vmcon import solve


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        (Problem(f=lambda x: x[0], df=lambda _: np.array([1])), -10.0),
        (Problem(f=lambda x: -x[0], df=lambda _: np.array([-1])), 10.0),
    ],
)
def test_vmcon_1d_10bounds(problem, expected):
    x, _, _, _ = solve(
        problem=problem,
        x=np.array([0.0]),
        lbs=np.array([-10]),
        ubs=np.array([10]),
        max_iter=100,
    )

    assert x.item() == expected


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        (
            Problem(f=lambda x: x[0] + x[1], df=lambda _: np.array([1, 1])),
            [-10.0, -20.0],
        ),
        (
            Problem(f=lambda x: -x[0] - x[1], df=lambda _: np.array([-1, -1])),
            [20.0, 10.0],
        ),
    ],
)
def test_vmcon_2d_1020bounds(problem, expected):
    x, _, _, _ = solve(
        problem=problem,
        x=np.array([0.0, 0.0]),
        lbs=np.array([-10, -20]),
        ubs=np.array([20, 10]),
        max_iter=100,
    )

    assert (x == expected).all()
