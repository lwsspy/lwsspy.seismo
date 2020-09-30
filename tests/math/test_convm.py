import numpy as np

from lwsspy.math.convm import convm


def test_convm():
    """Tests ``lwsspy.math.convm.py``"""

    u = np.array([-1, 2, 3, -2, 0, 1, 2])
    v = np.array([2, 4, -1, 1])
    solution = np.array([15, 5, -9, 7, 6, 7, -1])

    np.testing.assert_array_equal(convm(u,v), solution)
