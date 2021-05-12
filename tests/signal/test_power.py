
import numpy as np
import lwsspy as lpy


def test_power_l1():

    # Generate random data
    np.random.seed(12345)
    d = np.array([-1, -2, -3])
    s = np.array([-4, -5, -6])

    # Compute dlna
    powerl1 = lpy.power_l1(d, s)

    # Check if computation is ok.
    assert (powerl1-10*np.log10(6/15)) <= 1E-12


def test_power_l2():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])
    s = np.array([4, 5, 6])

    # Compute dlna
    powerl2 = lpy.power_l2(d, s)

    # Check if computation is ok.
    assert (powerl2-10*np.log10(14/77)) <= 1E-12
