import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from obspy.taup.tau import TauPyModel
import _pickle as pickle
from .. import base as lbase
from .. import plot as lplt


def plot_times(ax, arrivals, phase_colordict, zorderd, markersize=10):

    # extract the time/distance for each phase, and for each distance:
    for arrival in arrivals:
        dist = arrival.purist_distance % 360.0
        distance = arrival.distance
        if distance < 0:
            distance = (distance % 360)
        if abs(dist - distance) / dist > 1E-5:
            continue
        ax.plot(arrival.distance, arrival.time / 60, '.',
                label=arrival.name, color=phase_colordict[arrival.name],
                zorder=zorderd[arrival.name], markersize=markersize)

    return ax


def compute_traveltimes(source_depth, phase_list=[
        "P", "PP", "PKP", "PcP", "PPP",
        "S", "SS", "SKS", "ScS", "SSS"],
        min_degrees=0, max_degrees=180, npoints=50, model='ak135',
        verbose=True):

    # Get Model
    if not isinstance(model, TauPyModel):
        model = TauPyModel(model)

    # Prep phase dictionary
    adict = dict()

    # calculate the arrival times and plot vs. epicentral distance:
    degrees = np.linspace(min_degrees, max_degrees, npoints)
    notimes = []
    for degree in degrees:
        try:
            arrivals = model.get_travel_times(source_depth, degree,
                                              phase_list=phase_list)

            for arrival in arrivals:

                if arrival.name in adict:
                    adict[arrival.name]["time"].append(arrival.time)
                    adict[arrival.name]["distance"].append(arrival.distance)

                else:
                    adict[arrival.name] = dict()
                    adict[arrival.name]["time"] = [arrival.time]
                    adict[arrival.name]["distance"] = [arrival.distance]

        except ValueError:
            notimes.append(degree)

    if verbose:
        if len(notimes) == 1:
            tmpl = f"There was {len(notimes)} epicentral distance " \
                "without an arrival"
        else:
            tmpl = f"There were {len(notimes)} epicentral distances " \
                "without an arrival"
        print(tmpl)

    with open(os.path.join(lbase.DOWNLOAD_CACHE, "traveltimes.pkl"), 'wb') as f:
        pickle.dump(adict, file=f)

    return adict


def plot_traveltimes(
        source_depth, phase_list: Optional[None] = None,
        ax: Optional[Axes] = None,
        min_degrees=0, max_degrees=180, npoints=720, model='ak135',
        colordict: Optional[dict] = None,
        cmap='rainbow', legend=True, verbose=True, **kwargs):
    """Plots traveltimes into a given axes or creates axes to plot them into.
    While doing that it will also save a file of the computed traveltimes into
    a cache directory, so taht subsequent plotting is faster. 


    Parameters
    ----------
    source_depth : float
        source depth
    phase_list : Optional[list], optional
        list of phases to compute, by default None
    ax : Optional[Axes], optional
        axes to plot phases into, by default None
    min_degrees : int, optional
        start degrees, by default 0
    max_degrees : int, optional
        end degrees, by default 180
    npoints : int, optional
        number of points to compute, by default 360
    model : str, optional
        model to use for the computation, by default 'ak135'
    legend : bool, optional
        whether to compute the legend, by default True
    verbose : bool, optional
        whether to print some statements or not, by default True

    Returns
    -------
    Axes
        axes that the traveltimes where plotted into

    Notes
    -----

    A fairly complete phase list:

    .. code:: python

        phase_list= [
            "P", "PP", "PPP", "PcP", "Pdiff", "PKP", "PS", "PPS",
            "S", "SS", "SSS", "ScS", "Sdiff", "SKS", "SP", "SSP",
            "SSSS", "ScSScS",
        ]
    """

    if ax is None:
        ax = plt.gca()

    phase_file = os.path.join(lbase.DOWNLOAD_CACHE, "traveltimes.pkl")
    if os.path.exists(phase_file):
        with open(phase_file, 'rb') as f:
            adict = pickle.load(f)

        if phase_list is not None:
            if (all([
                True if _ph in adict else False
                    for _ph in phase_list]) is False):
                print("Need to compute traveltimes.")

                # adict = compute_traveltimes(
                #     source_depth, phase_list=phase_list,
                #     min_degrees=min_degrees, max_degrees=max_degrees,
                #     npoints=npoints, model=model)
            else:
                print("Traveltimes already computed.")
        else:
            print("Traveltimes already computed.")
    else:
        print("Need to compute traveltimes.")
        adict = compute_traveltimes(
            source_depth, phase_list=phase_list,
            min_degrees=min_degrees, max_degrees=max_degrees,
            npoints=npoints, model=model)

    if phase_list is None:
        phase_list = [k for k in adict.keys()]

    # NUmber of phases to plot
    Nph = len(phase_list)

    # Create color dictionary
    if colordict is None:
        colors = lplt.pick_colors_from_cmap(Nph, cmap=cmap)
        colordict = {ph: col for ph, col in zip(phase_list, colors)}

    # Create zorder dictionary so thatcertain phases are overwritten
    zorderd = {ph: Nph - _i for _i, ph in enumerate(phase_list)}

    # Plot phases into the axed

    for _ph in phase_list:

        ax.plot(adict[_ph]["distance"], np.array(adict[_ph]["time"])/60, '.',
                label=_ph, color=colordict[_ph], zorder=zorderd[_ph],
                **kwargs)

    if legend:
        # merge all arrival labels of a certain phase:
        handles, labels = ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        ax.legend(handles, labels, loc=2, numpoints=1)

    return ax
