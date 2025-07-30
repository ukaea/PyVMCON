"""Test individual units of the PyVMCON implementation."""

from dataclasses import dataclass

import numpy as np
import pytest

from pyvmcon.vmcon import _powells_gamma, _revise_B


@dataclass
class BRevisionAsset:
    """Test asset for testing B matrix revision."""

    B: np.ndarray
    ksi: np.ndarray
    eta: np.ndarray
    expected_return: np.ndarray


@pytest.mark.parametrize(
    "test_asset",
    [
        BRevisionAsset(
            B=np.identity(2),
            ksi=np.array([-0.66666666666666663, -0.83333333333333348]),
            eta=np.array([-1.3425925925925923, -1.7129629629629632]),
            expected_return=np.array(
                [
                    [1.3858727457706470, 0.50241291449459347],
                    [0.50241291449459347, 1.6536252239598812],
                ]
            ),
        ),
        BRevisionAsset(
            B=np.array(
                [
                    [2.1875467668036239, 1.4714414127452644],
                    [1.4714414127452644, 2.7501870672148332],
                ]
            ),
            ksi=np.array([-1.2385140125071988e-6, -6.1925700625482853e-7]),
            eta=np.array([-3.6205427119684330e-6, -3.5255433852299234e-6]),
            expected_return=np.array(
                [
                    [2.1875592084316073, 1.4714730232083293],
                    [1.4714730232083293, 2.7502368318126678],
                ]
            ),
        ),
    ],
)
def test_revise_B(test_asset):
    """Tests the hessian update implementation.

    Uses data from Example 1 of the NEA (Crane) to ensure PyVMCON agrees with that
    implementation to at least 14 decimal places.
    """
    new_B = _revise_B(test_asset.B, test_asset.ksi, test_asset.eta)

    # check symmetric
    np.testing.assert_array_equal(new_B, new_B.T)

    # check our revision agrees with NEA version of VMCON
    np.testing.assert_array_almost_equal(new_B, test_asset.expected_return, decimal=14)


@pytest.mark.parametrize(
    "test_asset",
    [
        BRevisionAsset(
            B=np.identity(2),
            ksi=np.array([-0.66666666666666663, -0.83333333333333348]),
            eta=np.array([-1.3425925925925923, -1.7129629629629632]),
            expected_return=np.array([-1.3425925925925923, -1.7129629629629632]),
        ),
        BRevisionAsset(
            B=np.array(
                [
                    [2.1875467668036239, 1.4714414127452644],
                    [1.4714414127452644, 2.7501870672148332],
                ]
            ),
            ksi=np.array([-1.2385140125071988e-6, -6.1925700625482853e-7]),
            eta=np.array([-3.6205427119684330e-6, -3.5255433852299234e-6]),
            expected_return=np.array(
                [-3.6205427119684330e-006, -3.5255433852299234e-006]
            ),
        ),
    ],
)
def test_powells_gamma(test_asset):
    """Tests the calculation of gamma (or eta) according to Powells paper.

    Uses data from Example 1 of the NEA (Crane) to ensure PyVMCON agrees with that
    implementation to at least 14 decimal places.
    """
    eta = _powells_gamma(test_asset.eta, test_asset.ksi, test_asset.B)

    # check our revision agrees with NEA version of VMCON
    np.testing.assert_array_almost_equal(eta, test_asset.expected_return, decimal=14)
