import numpy as np
import lwsspy as lpy


def test_dlna():

    # Generate random data
    np.random.seed(12345)
    d = np.random.random(100)

    # Compute dlna
    dlnA = lpy.dlna(d, d)

    # Check if computation is ok.
    assert abs(dlnA) <= 1E-12
