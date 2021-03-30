

import numpy as np
import lwsspy as lpy


def test_dlna():

    # Generate random data
    np.random.seed(12345)
    d = np.random.random(100)

    # Compute dlna
    max_cc, tshift = lpy.xcorr(d, d)

    # Check if computation is ok.
    assert abs(max_cc - 1.0) <= 1E-12
    assert abs(tshift) <= 1E-12
