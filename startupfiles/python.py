"""
This script is loaded at installation add the location of this script to the
.bashrc as 

export PYTHONSTARTUP=path/to/.startup.py

Then it will import all the cool function at python terminal startup.

Last modified: Lucas Sawade, 2020.09.15 01.00 (lsawade@princeton.edu)

"""

import os
if os.getenv("CONDA_DEFAULT_ENV", None) == "lwsspy":

    # External imports
    from numpy import *
    from matplotlib.pyplot import *
    from glob import glob

    # LWSSPY import
    from lwsspy.base.constants import *
    from lwsspy.geo import *
    from lwsspy.inversion import *
    from lwsspy.maps import *
    from lwsspy.math import *
    from lwsspy.pizza import *
    from lwsspy.plot import *
    from lwsspy.seismo import *
    from lwsspy.shell import *
    from lwsspy.signal import *
    from lwsspy.statistics import *
    from lwsspy.utils import *
    from lwsspy.weather import *

    # meshslice import
    # from meshslice import *

    # Updates plotting parameters
    # updaterc()  # in lwsspy

    # Set matplotlib tot interactivate mode
    ion()
