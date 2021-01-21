from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal
import lwsspy as lpy


def test_get_aspect():

    _, ax = plt.subplots()
    ax.set_aspect('equal')
    assert_almost_equal(lpy.get_aspect(ax), 1.)
    ax.set_aspect(10.0)
    assert_almost_equal(lpy.get_aspect(ax), 10.0)
