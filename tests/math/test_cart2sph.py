
import numpy as np
import lwsspy as lpy


def test_cart2sph1():
    """Tests ``lwsspy.math.geo2cart``"""

    # Input
    rtp = (1., (90.0-35.264389682754654)/180.0 * np.pi, 45.0/180.0 * np.pi)

    # Solution
    xyz = (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3))

    # Check
    np.testing.assert_almost_equal(rtp, lpy.cart2sph(*xyz))
