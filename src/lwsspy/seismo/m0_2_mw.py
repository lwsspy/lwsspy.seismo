import numpy as np


def m0_2_mw(m0):
    """Converts dyne-cm scalar moment M0 to Mw

    Parameters
    ----------
    m0 : numeric or array
        Scalar Moment

    Returns
    -------
    corresponding to input 
        Moment magnitude

    Notes
    -----

    :Reference: Kanamori, H. (1977)

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.01.07 19.00

    """

    mw = 2/3 * np.log10(m0) - 10.7

    return mw
