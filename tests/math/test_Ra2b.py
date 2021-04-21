import numpy as np
from numpy.testing import assert_array_almost_equal as aae
import lwsspy as lpy


def test_Ra2b():

    a = np.array((1, 0, 0))
    b = np.array((0, 1, 0))
    c = np.array((0, 0, 1))

    aae(a, lpy.Ra2b(b, a) @ b)
    aae(b, lpy.Ra2b(a, b) @ a)
    aae(a, lpy.Ra2b(c, a) @ c)
    aae(c, lpy.Ra2b(a, c) @ a)
    aae(b, lpy.Ra2b(c, b) @ c)
    aae(c, lpy.Ra2b(b, c) @ b)
