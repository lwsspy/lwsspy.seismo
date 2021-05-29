

import numpy as np
import lwsspy as lpy
from scipy.signal.windows import hann as taper


def test_dlna():

    # Generate random data
    np.random.seed(12345)
    d = np.random.random(100)

    # Compute dlna
    max_cc, tshift = lpy.xcorr(d, d)

    # Check if computation is ok.
    assert abs(max_cc - 1.0) <= 1E-12
    assert abs(tshift) <= 1E-12


def test_correct_index():

    times = np.linspace(0, 2*np.pi,  1000)
    data = np.sin(3*times) * taper(len(times))
    model = np.array([.5])

    def forward(a):
        return a * np.sin(3*times + 0.25*np.pi) * taper(len(times))

    istart, iend = 300, 700
    _, nshift = lpy.xcorr(data, forward(model))
    nshift
    istart_d, iend_d, istart_s, iend_s = lpy.correct_window_index(
        istart, iend, nshift, len(times))

    d_test_i, d_test_f = (340, 740)
    s_test_i, s_test_f = (300, 700)

    assert istart_s == s_test_i
    assert iend_s == s_test_f
    assert istart_d == d_test_i
    assert iend_d == d_test_f
