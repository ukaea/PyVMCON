import pytest
import numpy as np
from numpy.testing import assert_array_equal

from vmcon.types import coerce_vector


@pytest.mark.parametrize(
    "x, expected",
    (
        ([1], np.array([1])),
        ([1, 2, 3], np.array([1, 2, 3])),
        ([], np.array([])),
        (np.array([1]), np.array([1])),
        (np.array([1, 2, 3]), np.array([1, 2, 3])),
        (np.array([]), np.array([])),
        (np.array([[1]]), np.array([1])),
        (np.array([[1], [2], [3]]), np.array([1, 2, 3])),
    ),
)
def test_coerce_vector(x, expected):
    assert_array_equal(coerce_vector(x), expected)


@pytest.mark.parametrize(
    "x",
    (
        # code currently cannot identify ragged arrays
        # but I don't think it really needs to be able to
        # [1, [2, 3]],
        # [[1], [2, 3]],
        [[1, 2], [3, 4]],
        np.array([[1, 2], [3, 4]]),
        np.array([[], []]),
    ),
)
def test_coerce_vector_expect_error(x):
    with pytest.raises(ValueError):
        coerce_vector(x)
