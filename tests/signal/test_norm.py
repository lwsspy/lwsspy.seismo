
import numpy as np
import lwsspy as lpy


def test_norm2():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])
    s = np.array([4, 5, 6])

    # Compute dlna
    norm = lpy.norm2(d, s)

    # Check if computation is ok.
    assert (norm-13.5) <= 1E-12


def test_norm1():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])
    s = np.array([4, 5, 6])

    # Compute dlna
    norm = lpy.norm1(d, s)

    # Check if computation is ok.
    assert (norm-9.0) <= 1E-12
