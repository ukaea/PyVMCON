from typing import List, Union
import numpy as np
import numpy.typing as npt

Vector = Union[npt.NDArray[np.number], List[np.number], List[float]]
"""The types which could reasonably be an input into an objective 
function or constraint.

Notice that an objective function or constraint should NOT take a scalar as
an input; a 0-d numpy array should be used.
"""

NumpyVector = npt.NDArray[np.number]
"""An alias for a numpy array with elements 
of an arbitrary numeric type.

Note that a NumpyVector should have the shape (n,)"""


def coerce_vector(x: Vector, /) -> NumpyVector:
    """Coerces a Vector to a NumpyVector,
    ie an array which has shape (n,)

    Ensures that `x` is a vector, a numpy vector of shape (n,):
        [a, b, c]
    """
    if isinstance(x, list):
        # run lists through the coercer again
        # to check (and maybe broadcast) its shape
        return coerce_vector(np.array(x))

    # x is now assumed to be a numpy array
    if len(x.shape) == 1:
        return x
    elif len(x.shape) == 2 and x.shape[0] == 1:
        return x.squeeze(0)
    elif len(x.shape) == 2 and x.shape[1] == 1:
        return x.squeeze(1)

    raise ValueError(
        f"Unable to broadcast/convert {x.shape} to a column vector aka numpy array of shape (n, 1)"
    )
