"""

This directory contains function to reproduce figures that show parameters
for the Lamont Global CMT workflow, such as the data weighting and the
filtering


:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.orcopyleft/gpl.html)

Last Update: April 2020

"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
import lwsspy as lpy
from matplotlib.colors import ListedColormap

lplt.updaterc()

default_outputdir = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))),
    "gcmt3d"
)


def sac_cosine_taper(freqs, flimit):
    """ SAC style cosine taper.

    Args:
        freqs: vector to find a taper for
        flimit: corner frequencies

    Returns:

    """
    fl1, fl2, fl3, fl4 = flimit
    taper = np.zeros_like(freqs)

    # Beginning
    a = (fl1 <= freqs) & (freqs <= fl2)
    taper[a] = 0.5 * (1.0 - np.cos(np.pi * (freqs[a] - fl1) / (fl2 - fl1)))

    # Flat part
    b = (fl2 < freqs) & (freqs < fl3)
    taper[b] = 1.0

    # Ending
    c = (fl3 <= freqs) & (freqs <= fl4)
    taper[c] = 0.5 * (1.0 + np.cos(np.pi * (freqs[c] - fl3) / (fl4 - fl3)))

    return taper


def plot_taper(magnitude, *args, ax=None, outdir=None, **kwargs):
    """Plots single taper

    Parameters
    ----------
    ax : Axes, optional
        Axes to plot in, by default None
    """

    if ax is None:
        ax = plt.gca()

    # Magnitudes
    m = np.linspace(7.0, 8.0, 500)

    # Periods
    p = np.linspace(75.0, 450.0, 600)

    # Config values
    startcorners = [125, 150, 300, 350]
    endcorners = [200, 400]
    startmag = 7.0
    endmag = 8.0

    # Compute corners
    corners = lpy.gcmt3d.filter_scaling(
        startcorners, startmag, endcorners, endmag, magnitude)

    # Compute tapers
    taper = sac_cosine_taper(p, corners)

    # Plot tapers
    plt.plot(p, taper, *args, **kwargs)


def plot_tapers(ax=None, outdir=None):
    """Plots the weighting as a function of Period and magnitude."""

    # Magnitudes
    m = np.linspace(7.0, 8.0, 500)

    # Periods
    p = np.linspace(75.0, 450.0, 600)

    # Config values
    startcorners = [125, 150, 300, 350]
    endcorners = [200, 400]
    startmag = 7.0
    endmag = 8.0

    # Preallocate
    tapers = np.zeros((len(p), len(m)))
    corners = np.zeros((4, len(m)))

    for _i, mm in enumerate(m):
        # Compute corners
        corners[:, _i] = lpy.gcmt3d.filter_scaling(startcorners, startmag,
                                                   endcorners, endmag, mm)
        # Compute tapers
        tapers[:, _i] = sac_cosine_taper(p, corners[:, _i])

    if ax is None:
        ax = plt.gca()
    pc = ax.pcolormesh(p, m, tapers.T,
                       cmap=ListedColormap(plt.get_cmap(
                           'PuBu')(np.linspace(0, 1, 5)), N=5),
                       zorder=-10)
    lplt.nice_colorbar(pc, orientation="vertical", fraction=0.05, aspect=20,
                       pad=0.05)
    for _i in range(4):
        plt.plot(corners[_i, :], m, 'k')
    # plt.xlabel("Period [s]")
    # plt.ylabel(r"$M_w$")

    if outdir is not None:
        plt.savefig(os.path.join(outdir, "taperperiodmagnitude.png"), dpi=300)


def plot_weighting(outdir=None):
    """Plot Weighting of the different wavetypes."""

    # Momentmanitude vecotr
    mw = np.linspace(5.0, 9.0, 100)

    # Weights
    bodywaveweights = np.zeros_like(mw)
    surfacewaveweights = np.zeros_like(mw)
    mantlewaveweights = np.zeros_like(mw)

    # Corners
    bodywavecorners = np.zeros((len(mw), 4))
    surfacewavecorners = np.zeros((len(mw), 4))
    mantlewavecorners = np.zeros((len(mw), 4))

    for _i, m in enumerate(mw):

        P = lpy.gcmt3d.ProcessParams(m, 150000)
        P.determine_all()
        # assign
        bodywaveweights[_i] = P.bodywave_weight
        surfacewaveweights[_i] = P.surfacewave_weight
        mantlewaveweights[_i] = P.mantlewave_weight

        # Corners
        bodywavecorners[_i, :] = P.bodywave_filter
        surfacewavecorners[_i, :] = P.surfacewave_filter
        mantlewavecorners[_i, :] = P.mantlewave_filter

    fig = plt.figure(figsize=(8, 3))
    gs = fig.add_gridspec(1, 8)
    ax1 = fig.add_subplot(gs[0, :3])
    lplt.plot_label(ax1, "a", aspect=1, location=6, dist=0.025, box=False)
    plt.subplots_adjust(bottom=0.175, left=0.075, right=0.925, wspace=0.75)
    plt.plot(bodywaveweights, mw, "r", label="Body")
    plt.plot(surfacewaveweights, mw, "b:", label="Surface")
    plt.plot(mantlewaveweights, mw, "k", label="Mantle")
    plt.legend()
    plt.ylabel(r"$M_w$")
    plt.xlabel("Weight Contribution")

    ax2 = fig.add_subplot(gs[0, 3:], sharey=ax1)
    ax2.tick_params(labelleft=False)
    ax2.set_xlim((20.0, 450.0))
    lplt.plot_label(ax2, "Filter Corners",
                    location=4, dist=0.015, box=False)
    lplt.plot_label(ax2, "b", location=7, dist=0.025, box=False)

    for _i in range(4):
        if _i == 0:
            labelb = "Body"
            labels = "Surface"
            labelm = "Mantle"
        else:
            labels, labelb, labelm = None, None, None
        plt.plot(bodywavecorners[:, _i], mw, 'r', label=labelb)
        plt.plot(surfacewavecorners[:, _i], mw, 'b:', label=labels)
        plt.plot(mantlewavecorners[:, _i], mw, 'k', label=labelm)

    plt.xlabel("Period [s]")

    # Inset axis for the taper visualization.
    Mw = 7.25

    # Line for orientation in the main plot
    plt.plot([20, 450], [Mw, Mw], "k", marker="o", markersize=5, clip_on=False)

    axins = ax2.inset_axes([-0.05, 0.8, 0.4, 0.25])
    axins.set_rasterization_zorder(-5)

    P2 = lpy.gcmt3d.ProcessParams(Mw, 150000)
    P2.determine_all()

    # Corners
    bodywavecorners2 = P2.bodywave_filter
    surfacewavecorners2 = P2.surfacewave_filter
    mantlewavecorners2 = P2.mantlewave_filter

    # Plot taper
    p = np.linspace(20, 450, 1000)
    axins.plot(p, sac_cosine_taper(
        p, bodywavecorners2[::-1]), 'r', label="Body")

    # Plot corners
    for _c in bodywavecorners2[::-1]:
        axins.plot([_c, _c], [0, 1.25], 'b--', alpha=0.5, lw=0.5)

    for _c in mantlewavecorners2[::-1]:
        axins.plot([_c, _c], [0, 1.25], 'k--', alpha=0.5, lw=0.5)

    axins.plot(p, sac_cosine_taper(
        p, surfacewavecorners2[::-1]), 'b:', label="Surface")
    axins.plot(p, sac_cosine_taper(
        p, mantlewavecorners2[::-1]), 'k', label="Mantle")
    axins.plot([20, 450], [0, 0], "k", lw=0,
               marker="o", markersize=5, clip_on=False)
    axins.set_ylim((0, 1.25))
    axins.set_xlim((20.0, 450.0))
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    axins.set_title('Tapers', fontsize="x-small")
    # ax2.indicate_inset_zoom(axins)

    if outdir is not None:
        plt.savefig(os.path.join(outdir, "waveweighting.pdf"), dpi=900)
    else:
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="outdir", default=default_outputdir,
                        type=str, help="Output directory", required=False)
    args = parser.parse_args()

    # Plot tapers.
    # plot_tapers(outdir=args.outdir)

    plot_weighting(outdir=args.outdir)
