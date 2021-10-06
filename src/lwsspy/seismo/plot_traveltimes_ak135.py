from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os.path as p
from .. import base as lbase
from .. import utils as lutil
from .. import plot as lplt


def plot_traveltimes_ak135(ax: Union[None, Axes] = None,
                           labels: bool = True,
                           minutes: bool = True,
                           cmap: Union[str, None] = None,
                           labelkwargs: dict = {},
                           linekwargs: dict = {}):
    '''Loads ttbox ttc struct from matfile'''

    # Load ttc file
    filename = p.join(lbase.CONSTANT_DATA, 'ttc.mat')
    ttcfile = lutil.loadmat(filename)

    # Get vars
    N = ttcfile["ttc"]["anz"]  # Number of ttcurves
    ddeg = ttcfile["ttc"]["dangle"]  # delta degrees
    h = ttcfile["ttc"]["h"]  # Depth of the source
    modelname = ttcfile["ttc"]["name"]  # name of model used for raytracing
    ttcurves = ttcfile["ttc"]["ttc"]  # traveltime curves

    if ax is None:
        ax = plt.gca()

    if min:
        factor = 1.0/60.0
    else:
        factor = 1.0

    # colors = lplt.pick_colors_from_cmap(N, cmap=cmap)
    if cmap is None:
        colors = N * ['k']
    else:
        colors = lplt.pick_colors_from_cmap(N, cmap=cmap)

    for _curve, _color in zip(ttcurves, colors):
        # Wrap epicentral distances around 360
        _curve["d"] = np.mod(_curve["d"], 360)

        # find indeces that are larger than 180
        idx = np.where(_curve["d"] >= 180)

        # Fix values larger that 180
        _curve["d"][idx] = 360 - _curve["d"][idx]

        # Plot Traveltime curve
        plt.plot(_curve["d"], _curve["t"] * factor, label=_curve["p"],
                 c=_color, **linekwargs)

        # Plot text
        if labels:
            # Get npn-nan indeces of a traveltime curve
            firstindy = np.where(
                (~np.isnan(_curve["d"]))
                & (~np.isnan(_curve["t"])))

            if np.size(firstindy) != 0:
                # Get first non-nan value
                firstindy = firstindy[0][0]

                # Manual offset overwrite
                if _curve["p"] == "SKS":
                    xoffset = -25.0
                else:
                    xoffset = 0.0

                # Plot label into figure
                if _curve["d"][firstindy] > 90:
                    halign = 'right'
                else:
                    halign = 'left'

                t = ax.text(
                    _curve["d"][firstindy] + xoffset,
                    _curve["t"][firstindy] * factor,
                    _curve["p"], horizontalalignment=halign, c=_color,
                    verticalalignment='bottom', zorder=5,
                    ** labelkwargs)
