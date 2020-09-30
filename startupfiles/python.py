"""
This script is loaded at installation add the location of this script to the
.bashrc as 

export PYTHONSTARTUP=path/to/.startup.py

Then it will import all the cool function at python terminal startup.

Last modified: Lucas Sawade, 2020.09.15 01.00 (lsawade@princeton.edu)

"""

import os
if os.environ["CONDA_DEFAULT_ENV"] == "lwsspy":

    # External imports
    from numpy import *
    from matplotlib.pyplot import *
    from glob import glob

    # LWSSPY import
    from lwsspy import *

    # Updates plotting parameters
    updaterc()  # in lwsspy

    # Set matplotlib tot interactivate mode
    ion()
