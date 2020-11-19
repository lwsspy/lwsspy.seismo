
import numpy as np
import lwsspy as lpy


def test_geo2cart1():
    """Tests ``lwsspy.math.geo2cart``"""

    # Input
    rtp = (1., 35.264389682754654, 45.0)

    # Solution
    xyz = (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3))

    # Check
    np.testing.assert_almost_equal(xyz, lpy.geo2cart(*rtp))


def test_geo2cart2():
    """Tests ``lwsspy.math.geo2cart``"""

    # Input
    rtp = (1., 35.264389682754654, -45.0)

    # Solution
    xyz = (1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3))

    # Check
    np.testing.assert_almost_equal(xyz, lpy.geo2cart(*rtp))


def test_geo2cart3():
    """Tests ``lwsspy.math.geo2cart``"""

    # Input
    rtp = (1., 35.264389682754654, -135.0)

    # Solution
    xyz = (-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3))

    # Check
    np.testing.assert_almost_equal(xyz, lpy.geo2cart(*rtp))


def test_geo2cart4():
    """Tests ``lwsspy.math.geo2cart``"""

    # Input
    rtp = (1., 35.264389682754654, 135.0)

    # Solution
    xyz = (-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3))

    # Check
    np.testing.assert_almost_equal(xyz, lpy.geo2cart(*rtp))
