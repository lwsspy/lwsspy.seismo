
import lwsspy as lpy
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_magnitude():

    assert lpy.magnitude(230.03424235) == 2
    assert lpy.magnitude(20.32) == 1
    assert lpy.magnitude(0.0023) == -3
    assert lpy.magnitude(2302342) == 6
    assert lpy.magnitude(0) == 0
    assert_array_almost_equal(lpy.magnitude((245, 0, 0.00004)), (2, 0, -5))
