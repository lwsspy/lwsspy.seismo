
from ast import Call
from curses import has_key
from enum import Flag
from re import S
from signal import raise_signal
from time import time
from typing import Callable, Optional, Union, List, Iterable
from unittest.mock import NonCallableMagicMock
from matplotlib import gridspec
from more_itertools import only
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from obspy import Trace, Stream, UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from cartopy.crs import AzimuthalEquidistant, Orthographic, Geodetic, \
    PlateCarree
from pyflex.window import Window
from .source import CMTSource, plot_beach
from lwsspy.plot.plot_label import plot_label
from lwsspy.plot.axes_from_axes import axes_from_axes
from lwsspy.plot.geo2axes import geo2axes
from lwsspy.plot.get_aspect import get_aspect
from lwsspy.maps.midpoint import geomidpointv
from lwsspy.maps.reckon import reckon
from lwsspy.maps.plot_map import plot_map
from lwsspy.plot.get_limits import get_limits
from lwsspy.math.envelope import envelope
from lwsspy.signal.xcorr import xcorr, correct_window_index
from lwsspy.signal.norm import dnorm2, norm2
from lwsspy.signal.dlna import dlna
from matplotlib import rcParams


# def get_toffset(
#         tsample: int, dt: float, t0: UTCDateTime, origin: UTCDateTime) -> float:
#     """Computes the time of a sample with respect to origin time

#     Parameters
#     ----------
#     tsample : int
#         sample on trace
#     dt : float
#         sample spacing
#     t0 : UTCDateTime
#         time of the first sample
#     origin : UTCDateTime
#         origin time

#     Returns
#     -------
#     float
#         Time relative to origin time
#     """

#     # Second on trace
#     trsec = (tsample*dt)
#     return (t0 + trsec) - origin

import typing as tp
from obspy.geodetics.base import gps2dist_azimuth

# Create String


def NS(latitude):
    return 'N' if latitude > 0 else 'S'


def EW(longitude):
    return 'E' if longitude > 0 else 'W'


def cmtheader(network, station, slat=None, slon=None, cmt=None, periodrange=None):

    header = f"{network}-{station}"

    m2deg = 360/(2*np.pi*6371000.0)

    # Put Station location
    if (slat is not None) and (slon is not None):

        header += (
            f" - $\\theta$={np.abs(slat):.2f}{NS(slat)}, "
            f"$\\phi$={np.abs(slon):.2f}{EW(slon)}")

    # Put event info
    if cmt is not None:
        clat = cmt.latitude
        clon = cmt.longitude

        # Add cmt info
        header += (
            f"\n{cmt.eventname} - "
            f"{cmt.cmt_time.strftime('%Y/%m/%d %H:%M:%S')}, "
            f"$\\theta$={np.abs(clat):.2f}{NS(clat)}, "
            f"$\\phi$={np.abs(clon):.2f}{EW(clon)}, "
            f"$h$={cmt.depth_in_m/1000.0:5.1f}"
        )
    else:
        clat, clon = None, None

    # Put Station/event geometry
    if (slat is not None) and (slon is not None) and (clat is not None) and (clon is not None):

        # Get some geographical data
        dist, az, baz = gps2dist_azimuth(clat, clon, slat, slon)

        header += (
            f"\n$\\Delta$={m2deg*dist:6.2f}, "
            f"$\\alpha$={az:6.2f}, "
            f"$\\beta$={baz:6.2f}")

    # Put Period range
    if periodrange is not None:
        header += f", P: {int(periodrange[0]):d}-{int(periodrange[1]):d}s"

    return header


def cmtstring(
        cmt: CMTSource, initcmt: tp.Optional[CMTSource] = None, labels: bool = True,
        labelsonly: bool = False):

    if labelsonly:
        labels = True

    # Define labels
    Mw = f" Mw = " if labels else ""
    S = f"  S = " if labels else ""
    D = f"  D = " if labels else ""
    R = f"  R = " if labels else ""
    Ax = f" Ax = " if labels else ""
    T = f"  T = " if labels else ""
    B = f"  B = " if labels else ""
    P = f"  P = " if labels else ""
    eps = f"eps = " if labels else ""
    t = f"  t = " if labels else ""
    th = f" th = " if labels else ""
    ph = f" ph = " if labels else ""
    h = f"  h = " if labels else ""
    line = "------" if labels else ""
    if labelsonly:
        return f"{Mw}\n{S}\n{D}\n{R}\n\n{Ax}\n{T}\n{B}\n{P}\n{eps}\n\n{t}\n{th}\n{ph}\n{h}"

    # Check whether diff
    if isinstance(initcmt, CMTSource):
        diff = True
        dcmt = cmt - initcmt
    else:
        diff = False

    # Strike dip rake and plunge axes
    ss, ds, rs = cmt.sdr
    ss = np.round(ss).astype(int)
    ds = np.round(ds).astype(int)
    rs = np.round(rs).astype(int)

    # Get Plunge and
    _, _, _, _, _, _, plungs, azims = cmt.eqpar
    plungs = np.round(plungs).astype(int)
    azims = np.round(azims).astype(int)

    # Get DC component
    string = (
        f"{Mw}{cmt.moment_magnitude:>11.2f}\n"
        f"{S}{ss[0]:>4d} | {ss[1]:>4d}\n"
        f"{D}{ds[0]:>4d} | {ds[1]:>4d}\n"
        f"{R}{rs[0]:>4d} | {rs[1]:>4d}\n"
        f"{line}------------\n"
        f"{Ax} Azi,    Pl\n"
        f"{T}{azims[0]:>4d}, {plungs[0]:>5d}\n"
        f"{B}{azims[1]:>4d}, {plungs[1]:>5d}\n"
        f"{P}{azims[2]:>4d}, {plungs[2]:>5d}\n"
        f"{eps}{cmt.decomp(dtype='eps_nu')[0]:11.3f}\n"
        f"{line}------------\n"
    )

    if diff:
        string += (
            f"{dcmt.time_shift:>10.3f}d\n"
            f"{dcmt.latitude:>10.3f}d\n"
            f"{dcmt.longitude:>10.3f}d\n"
            f"{dcmt.depth_in_m/1000:>10.3f}d"
        )
    else:
        string += (
            f"{t}{cmt.time_shift:>10.3f} \n"
            f"{th}{np.abs(cmt.latitude):>10.3f}{NS(cmt.latitude)}\n"
            f"{ph}{np.abs(cmt.longitude):>10.3f}{EW(cmt.longitude)}\n"
            f"{h}{cmt.depth_in_m/1000:>10.3f} "
        )

    return string


def az_arrow(ax, x, y, r, angle, *args, **kwargs):

    # Get dx dependent on the radius
    dx = r*np.sin(angle/180*np.pi)
    dy = r*np.cos(angle/180*np.pi)

    # Make "Annotation"
    ax.arrow(x, y, dx, dy, *args, **kwargs, transform=ax.transAxes,
             clip_on=False)


# def az_arrow(ax, x1, y1, x2, y2, *args, **kwargs):
#     q = ax.quiver(
#         np.array([x1]), np.array([y1]), np.array([x2-x1]), np.array([y2-y1]),
#         # angles='xy', scale_units='xy',
#         *args, **kwargs)
#     return q


# def az_arrow_bu(ax, x, y, r, angle, *args, **kwargs):
#     dx = r*np.sin(angle/180*np.pi)
#     dy = r*np.cos(angle/180*np.pi)
#     ax.arrow(x, y, dx, dy, *args, **kwargs)


# def az_arrow_r(ax, x, y, r, angle, *args, **kwargs):
#     dx = r*np.sin(angle/180*np.pi)
#     dy = r*np.cos(angle/180*np.pi)
#     ax.arrow(x+dx, y+dy, -dx, -dy, *args, **kwargs)


def get_mcsta(d, s, dt, npts, leftidx, rightidx, taper=1.0):

    # Shorten left and right
    l, r = leftidx, rightidx

    # Get window data
    wd = d[l:r]
    ws = s[l:r]

    # winleft = get_toffset(
    #     l, dt, win.time_of_first_sample, event.origin_time)
    # winright = get_toffset(
    #     win.right, dt, win.time_of_first_sample, event.origin_time)

    # Measurements
    max_cc_value, nshift = xcorr(wd*taper, ws*taper)

    # Get fixed window indeces.
    istart, iend = l, r
    istart_d, iend_d, istart_s, iend_s = correct_window_index(
        istart, iend, nshift, npts)
    wd_fix = d[istart_d:iend_d]
    ws_fix = s[istart_s:iend_s]

    # Get measurements
    m = dnorm2(wd, ws, w=taper)/norm2(wd, w=taper)
    c = max_cc_value
    s = np.sum(taper * wd_fix * ws_fix)/np.sum(taper * ws_fix ** 2)
    t = nshift * dt
    a = dlna(wd_fix, ws_fix, w=taper)

    return m, c, s, t, a


def plot_seismogram(obsd: Trace,
                    synt: Optional[Trace] = None,
                    cmtsource: Optional[CMTSource] = None,
                    tag: Optional[str] = None):
    station = obsd.stats.station
    network = obsd.stats.network
    channel = obsd.stats.channel
    location = obsd.stats.location

    trace_id = f"{network}.{station}.{location}.{channel}"

    # Times and offsets computed individually, since the grid search applies
    # a timeshift which changes the times of the traces.
    if cmtsource is None:
        offset = 0
        if isinstance(synt, Trace):
            offset_synt = 0
    else:
        offset = obsd.stats.starttime - cmtsource.cmt_time
        if isinstance(synt, Trace):
            offset_synt = synt.stats.starttime - cmtsource.cmt_time

    times = [offset + obsd.stats.delta * i for i in range(obsd.stats.npts)]
    if isinstance(synt, Trace):
        times_synt = [offset_synt + synt.stats.delta * i
                      for i in range(synt.stats.npts)]

    # Figure Setup
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(211)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95)

    ax1.plot(times, obsd.data, color="black", linewidth=0.75,
             label="Observed")
    if isinstance(synt, Trace):
        ax1.plot(times_synt, synt.data, color="red", linewidth=0.75,
                 label="Synthetic")
    ax1.set_xlim(times[0], times[-1])
    ax1.legend(loc='upper right', frameon=False, ncol=1, prop={'size': 11})
    ax1.tick_params(labelbottom=False, labeltop=False)

    # Setting top left corner text manually
    if isinstance(tag, str):
        label = f"{trace_id}\n{tag.capitalize()}"
    else:
        label = f"{trace_id}"
    plot_label(ax1, label, location=1, dist=0.005, box=False)

    # plot envelope
    ax2 = plt.subplot(212)
    ax2.plot(times, envelope(obsd.data), color="black",
             linewidth=1.0, label="Observed")
    if isinstance(synt, Trace):
        ax2.plot(times_synt, envelope(synt.data), color="red", linewidth=1,
                 label="Synthetic")
    ax2.set_xlim(times[0], times[-1])
    ax2.set_xlabel("Time [s]", fontsize=13)
    plot_label(ax2, "Envelope", location=1, dist=0.005, box=False)
    if isinstance(synt, Trace):
        try:
            for win in obsd.stats.windows:
                left = times[win.left]
                right = times[win.right]
                re1 = Rectangle((left, ax1.get_ylim()[0]), right - left,
                                ax1.get_ylim()[1] - ax1.get_ylim()[0],
                                color="blue", alpha=0.25, zorder=-1)
                ax1.add_patch(re1)
                re2 = Rectangle((left, ax2.get_ylim()[0]), right - left,
                                ax2.get_ylim()[1] - ax2.get_ylim()[0],
                                color="blue", alpha=0.25, zorder=-1)
                ax2.add_patch(re2)
        except Exception as e:
            print(e)

    return fig


def plot_seismograms(
        obsd: Optional[Trace] = None,
        synt: Optional[Trace] = None,
        syntf: Optional[Trace] = None,
        cmtsource: Optional[CMTSource] = None,
        tag: Union[str, None] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        processfunc: Callable = lambda x: x.data,
        annotations: bool = True,
        labelbottom: bool = False,
        labelobsd: str = None,
        labelsynt: str = None,
        labelnewsynt: str = None,
        legend: bool = True,
        windows: bool = True,
        window_list: List[dict] = None,
        timescale: float = 1.0,
        obsdcolor='k',
        syntcolor='r',
        newsyntcolor='b',
        xlim_in_seconds: Optional[Iterable[float]] = None):

    if ax is None:
        ax = plt.gca()

    # Check whether there are traces to plot
    plotobsd = isinstance(obsd, Trace)
    plotsynt = isinstance(synt, Trace)
    plotsyntf = isinstance(syntf, Trace)

    if plotobsd:
        station = obsd.stats.station
        network = obsd.stats.network
        channel = obsd.stats.channel
        location = obsd.stats.location
    elif plotsynt:
        station = synt.stats.station
        network = synt.stats.network
        channel = synt.stats.channel
        location = synt.stats.location
    elif plotsyntf:
        station = syntf.stats.station
        network = syntf.stats.network
        channel = syntf.stats.channel
        location = syntf.stats.location
    else:
        raise ValueError("You gotta at least have a single trace, boi.")

    if not labelobsd:
        labelobsd = 'Obs'
    if not labelsynt:
        labelsynt = 'Syn'
    if not labelnewsynt:
        labelnewsynt = 'New Syn'

    # Create trace ID to put a label on the plot.
    trace_id = f"{network}.{station}.{location}.{channel}"

    # Times and offsets computed individually, since the grid search applies
    # a timeshift which changes the times of the traces.
    if cmtsource is None:
        offset_obsd = 0
        offset_synt = 0
        offset_syntf = 0
    else:
        if plotobsd:
            offset_obsd = obsd.stats.starttime - cmtsource.cmt_time
        if plotsynt:
            offset_synt = synt.stats.starttime - cmtsource.cmt_time
        if plotsyntf:
            offset_syntf = syntf.stats.starttime - cmtsource.cmt_time

    # Alpha values
    alpha = 1.0 if windows is False else 0.5
    maxdisp = 0.0
    if plotobsd:

        # Times
        times_obsd = offset_obsd + obsd.stats.delta * \
            np.arange(obsd.stats.npts)

        # Get data
        pobsd = processfunc(obsd)

        # Timeband data
        if xlim_in_seconds:
            obsdpos = np.where(
                (xlim_in_seconds[0] <= times_obsd) &
                (times_obsd <= xlim_in_seconds[1]))[0]
            times_obsd = times_obsd[obsdpos]
            pobsd = pobsd[obsdpos]

        # Plot seismograms
        ax.plot(np.array(times_obsd)/timescale, pobsd, color=obsdcolor,
                linewidth=0.75, label=labelobsd, alpha=alpha)

    if plotsynt:

        # Times
        times_synt = offset_synt + synt.stats.delta * \
            np.arange(synt.stats.npts)

        # Get data
        psynt = processfunc(synt)

        # Timeband data
        if xlim_in_seconds:
            syntpos = np.where(
                (xlim_in_seconds[0] <= times_synt) &
                (times_synt <= xlim_in_seconds[1]))[0]
            times_synt = times_synt[syntpos]
            psynt = psynt[syntpos]

        # Plot seismograms
        ax.plot(np.array(times_synt)/timescale, psynt, color=syntcolor,
                linewidth=0.75, label=labelsynt, alpha=alpha)

    if plotsyntf:

        # Times
        times_syntf = offset_syntf + syntf.stats.delta * \
            np.arange(syntf.stats.npts)

        # Get data
        psyntf = processfunc(syntf)

        # Timeband data
        if xlim_in_seconds:
            syntfpos = np.where(
                (xlim_in_seconds[0] <= times_syntf) &
                (times_syntf <= xlim_in_seconds[1]))[0]
            times_syntf = times_syntf[syntfpos]
            psyntf = psyntf[syntfpos]

        # Plot seismograms
        ax.plot(np.array(times_syntf)/timescale, psyntf, color=newsyntcolor,
                linewidth=0.75, label=labelnewsynt, alpha=alpha)

    if legend:
        ax.legend(loc='lower right', frameon=False, ncol=1,
                  prop=dict(size=11, family='monospace'),
                  bbox_to_anchor=(1., 1.), borderaxespad=0.0)
    ax.tick_params(labelbottom=labelbottom, labeltop=False)

    if xlim_in_seconds:
        xmin = xlim_in_seconds[0]/timescale
        xmax = xlim_in_seconds[1]/timescale

        # Set x axis limits
        ax.set_xlim(xmin, xmax)

    # Get data limits
    _, _, ymin, ymax = get_limits(ax)

    # Get maximum displacement in terms of ydata
    maxdisp = np.max(np.abs((ymin, ymax)))

    # Plot windows
    if plotobsd and (plotsynt or plotsyntf) and (windows):

        # If custom windows are given check those first,
        # if not check whether obsd.stats are given
        # finally accept that there aren't any window :(
        if (window_list is not None) and (cmtsource is not None):
            customwindows = True
            window_list = window_list
        elif hasattr(obsd.stats, 'windows'):
            customwindows = False
            window_list = obsd.stats.windows
        else:
            customwindows = False
            window_list = []
        # print("trying to print windows")
        try:
            for _i, win in enumerate(window_list):

                # Get window length from custom or trace attached window
                if customwindows:
                    if isinstance(win, dict):
                        # Compute actual startime
                        left = UTCDateTime(win['absolute_starttime']) \
                            - cmtsource.cmt_time
                        right = UTCDateTime(win['absolute_endtime']) \
                            - cmtsource.cmt_time

                    elif isinstance(win, Window):
                        # Compute actual startime
                        left = win.absolute_starttime - cmtsource.cmt_time
                        right = win.absolute_endtime - cmtsource.cmt_time

                    # Get left and right indeces
                    leftidx = np.argmin(np.abs(times_obsd-left))
                    rightidx = np.argmin(np.abs(times_obsd-right))

                    # Scale the time from cmt_time
                    left /= timescale
                    right /= timescale

                else:
                    if xlim_in_seconds:
                        if win.left in obsdpos and win.right in obsdpos:
                            wleft = obsdpos.tolist().index(win.left)
                            wright = obsdpos.tolist().index(win.right)
                            left = times_obsd[wleft]/timescale
                            right = times_obsd[wright]/timescale
                        else:
                            continue
                    else:
                        left = times_obsd[win.left]/timescale
                        right = times_obsd[win.right]/timescale

                    # Get left and right indeces
                    leftidx = win.left
                    rightidx = win.right

                # Plot windows
                if plotobsd:
                    ax.plot(
                        np.array(times_obsd[leftidx:rightidx])/timescale,
                        pobsd[leftidx:rightidx], color=obsdcolor,
                        linewidth=1.0, label=labelobsd if _i == 0 else None)
                if plotsynt:
                    ax.plot(
                        np.array(times_synt[leftidx:rightidx])/timescale,
                        psynt[leftidx:rightidx], color=syntcolor,
                        linewidth=1.0, label=labelsynt if _i == 0 else None)
                if plotsyntf:
                    ax.plot(
                        np.array(times_syntf[leftidx:rightidx])/timescale,
                        psyntf[leftidx:rightidx], color=newsyntcolor,
                        linewidth=1.0, label=labelnewsynt if _i == 0 else None)

                # Create Rectangle
                re1 = Rectangle((left, -maxdisp),
                                right - left, + 2*maxdisp,
                                color="blue", alpha=0.1, zorder=-1)

                # Add rectangle to Axes
                ax.add_patch(re1)

                # Add annotations
                if annotations:
                    trans = transforms.blended_transform_factory(
                        ax.transData, ax.transAxes)

                    if hasattr(obsd.stats, 'tapers'):
                        taper = obsd.stats.tapers[_i]
                    else:
                        taper = 1.0

                    m, c, s, t, a = get_mcsta(
                        pobsd, psynt, obsd.stats.delta, obsd.stats.npts,
                        leftidx, rightidx, taper=taper)

                    # Create strings
                    addstring0 = (
                        f"  {m:5.2f}"
                        f"\n  {c:5.2f}"
                        f"\n  {s:5.2f}"
                        f"\n  {t:5.2f}"
                        f"\n  {a:5.2f}"
                    )

                    # va = 'top' if (_i % 2) == 0 else 'bottom'
                    # y = 0.98 if (_i % 2) == 0 else 0.02
                    va = 'top'
                    y = 0.15

                    ax.text(left, y, "M:\nC:\nS:\nT:\nA:", transform=trans,
                            fontdict=dict(size="xx-small", family='monospace'),
                            horizontalalignment='left',
                            verticalalignment=va)

                    ax.text(left, y, addstring0, transform=trans,
                            fontdict=dict(size="xx-small", family='monospace'),
                            horizontalalignment='left',
                            verticalalignment=va, color=syntcolor)

                    if plotsyntf:
                        m, c, s, t, a = get_mcsta(
                            pobsd, psyntf, obsd.stats.delta, obsd.stats.npts,
                            leftidx, rightidx, taper=taper)

                        # Create strings
                        addstring1 = (
                            f"        {m:5.2f}"
                            f"\n        {c:5.2f}"
                            f"\n        {s:5.2f}"
                            f"\n        {t:5.2f}"
                            f"\n        {a:5.2f}"
                        )

                        # Print second string
                        ax.text(left, y, addstring1, transform=trans,
                                fontdict=dict(size="xx-small",
                                              family='monospace'),
                                horizontalalignment='left',
                                verticalalignment=va, color=newsyntcolor)

        except Exception as e:
            print(e)

    # Change fontsize
    fontsizech = 'medium'
    fontsize = 'x-small'

    # Create infostring
    channelstring = f"{channel}-{location}"
    dispstring = f"{maxdisp*1e6:.2f}$\mu$"

    # Plot base info
    plot_label(ax, channelstring, location=8, box=False, fontfamily='monospace',
               dist=0.01, fontsize=fontsizech)

    if plotobsd:
        if hasattr(obsd.stats, 'weights'):
            dispstring = "\n\n" + \
                f"W: {obsd.stats.weights: 6.4f}\n" + dispstring
        else:
            dispstring = "\n\n" + dispstring
    else:
        dispstring = "\n\n" + dispstring
        # plot_label(ax, weightstring, location=8, box=False,
        #            fontfamily='monospace', dist=0.01, fontsize=fontsize)

    plot_label(ax, dispstring, location=8, box=False,
               fontfamily='monospace', dist=0.01, fontsize=fontsize)

    # Add Measurements
    if plotobsd and plotsynt and plotsyntf:
        # Misfit and Cross correlation
        FS = np.sum((psynt-pobsd)**2)/np.sum(pobsd**2)
        CS, TS = xcorr(pobsd, psynt)
        SS = np.sum(np.roll(psynt, TS) * pobsd)/np.sum(psynt**2)

        FF = np.sum((psyntf-pobsd)**2)/np.sum(pobsd**2)
        CF, TF = xcorr(pobsd, psyntf)
        SF = np.sum(np.roll(psyntf, TF) * pobsd)/np.sum(psyntf**2)
        plot_label(ax, "\n\n\n\nM:\n\nC:\n\nS:", location=8, box=False,
                   fontfamily='monospace', dist=0.01, fontsize=fontsize)

        # Create strings
        addstring0 = (
            f"\n\n\n\n   {FS:.3f}"
            f"\n\n   {CS:.3f}"
            f"\n\n   {SS:.3f}"
        )
        addstring1 = (
            f"\n\n\n\n\n   {FF:.3f}"
            f"\n\n   {CF:.3f}"
            f"\n\n   {SF:.3f}"
        )

        plot_label(ax, addstring0, location=8, box=False,
                   fontfamily='monospace', fontsize=fontsize,
                   dist=0.01, color=syntcolor)

        plot_label(ax, addstring1, location=8, box=False,
                   fontfamily='monospace', fontsize=fontsize,
                   dist=0.01, color=newsyntcolor)

    elif plotobsd and plotsynt and (plotsyntf is False):
        # Misfit and Cross correlation
        FS = np.sum((psynt-pobsd)**2)/np.sum(pobsd**2)
        CS, TS = xcorr(pobsd, psynt)
        SS = np.sum(np.roll(psynt, TS) * pobsd)/np.sum(psynt**2)

        plot_label(ax, "\n\n\n\nM:\nC:\nS:", location=8, box=False,
                   fontfamily='monospace', dist=0.01, fontsize=fontsize)

        # Create strings
        addstring0 = (
            f"\n\n\n\n   {FS:.3f}"
            f"\n   {CS:.3f}"
            f"\n   {SS:.3f}"
        )

        plot_label(ax, addstring0, location=8, box=False,
                   fontfamily='monospace', dist=0.01, color=syntcolor,
                   fontsize=fontsize)


def plot_seismogram_by_station(
        network: str,
        station: str,
        obsd: Optional[Stream] = None,
        synt: Optional[Stream] = None,
        newsynt: Optional[Stream] = None,
        obsdcmt: Optional[CMTSource] = None,
        syntcmt: Optional[CMTSource] = None,
        newsyntcmt: Optional[CMTSource] = None,
        tag: Optional[str] = None,
        compsystem: str = "ZRT",
        location: str = "00",
        logger: Callable = print,
        processfunc: Callable = lambda x: x.data,
        annotations: bool = False,
        labelobsd: Optional[str] = None,
        labelsynt: Optional[str] = None,
        labelnewsynt: Optional[str] = None,
        periodrange: Optional[list] = None,
        windows: bool = False,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        window_dict: Optional[dict] = None,
        map: bool = False,
        letters: bool = False,
        plot_beach: bool = False,
        timescale: float = 3600.0,
        midpointmap: bool = False,
        obsdcolor='k',
        syntcolor=(0.9, 0.2, 0.2),
        newsyntcolor=(0.2, 0.2, 0.8),
        pdfmode: bool = False,
        xlim_in_seconds: Optional[Iterable[float]] = None,
        eventdetails: bool = True):

    # Get and set font family
    defff = matplotlib.rcParams['font.family']
    matplotlib.rcParams.update({'font.family': 'monospace'})

    # Some flag corrections
    # Annotations can only be printed if windows are to be printed as well
    annotations = annotations if windows else False

    Ncomp = len(compsystem)

    if obsd is not None:
        obs = obsd.select(network=network, station=station)
        if len(obs) == 0:
            logger(f"{network}.{station} not in observed stream.")
            obs = None
    else:
        obs = None

    if synt is not None:
        syn = synt.select(network=network, station=station)
        if len(syn) == 0:
            logger(f"{network}.{station} not in synthetic stream.")
            syn = None
    else:
        syn = None

    if newsynt is not None:
        newsyn = newsynt.select(network=network, station=station)
        if len(newsyn) == 0:
            logger(
                f"{network}.{station} not in new synthetic stream.")
            newsyn = None
    else:
        newsyn = None

    if not labelobsd:
        labelobsd = 'Obs'
    if not labelsynt:
        labelsynt = 'Syn'
    if not labelnewsynt:
        labelnewsynt = 'New Syn'

    # Figure Setup
    components = [x for x in compsystem]
    nrows = len(components)

    # check whether any cmts should be plotted
    if (obsdcmt is not None) or (syntcmt is not None) or (newsyntcmt is not None):
        cmtcomp = True
    else:
        cmtcomp = False

    # Whether to have the left colum for map info etc.
    if map or cmtcomp:
        twocol = True
        ncols = 2
        width_ratios = [1, 4]
    else:
        twocol = False
        ncols = 1
        width_ratios = [1]

    # Total figure width
    # left = 0.2 if twocol else 0.1
    leftfig = 1.0 if twocol else 0.0

    # Figure setup depending on different parameters
    figfactor = 1.75  # if annotations else 1.3
    hspace = 0.3 if annotations else 0.1
    bottom = 0.15  # if annotations else 0.1

    # Figure and Gridspec setups
    fig = plt.figure(figsize=(15+leftfig, 1.0 + figfactor*Ncomp))
    outer = GridSpec(
        nrows=1, ncols=ncols, width_ratios=width_ratios,
        wspace=0.01)

    if ncols == 2:
        # Map and cmt axes
        GSL = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0], hspace=0.1, height_ratios=[1, 2])

        # Seismogram axes
        GSR = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[1], hspace=hspace)
    else:
        # Seismogram axes
        GSR = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[0], hspace=hspace)

    # Seismogram axes
    axes = [fig.add_subplot(GSR[i]) for i in range(nrows)]

    # Adjusting the figure size
    plt.subplots_adjust(top=0.85 - 0.05 * (3-Ncomp), hspace=hspace,
                        bottom=bottom + 0.05 * (3-Ncomp))

    if plot_beach and cmtcomp:

        axcmts = []
        cmtcolors = []
        cmtlabels = []
        if obsdcmt is not None:
            axcmts.append(obsdcmt)
            cmtcolors.append(obsdcolor)
            cmtlabels.append('obsd')
        if syntcmt is not None:
            axcmts.append(syntcmt)
            cmtcolors.append(syntcolor)
            cmtlabels.append('synt')
        if newsyntcmt is not None:
            axcmts.append(newsyntcmt)
            cmtcolors.append(newsyntcolor)
            cmtlabels.append('newsynt')

        beachgs = gridspec.GridSpecFromSubplotSpec(
            1, len(axcmts), subplot_spec=GSL[1])

        # plot beaches in the axes anyways
        widthperdpi = 80/140
        if pdfmode:
            beachwidth = widthperdpi * 72
        else:
            beachwidth = widthperdpi*rcParams['figure.dpi']

        if len(axcmts) == 3:
            beachfactor = 1
            cmtstringfontsize = 'xx-small'
            y = 0.75
        elif len(axcmts) == 2:
            beachfactor = 1.5
            cmtstringfontsize = 'x-small'
            y = 0.85
        else:
            beachfactor = 1.5
            cmtstringfontsize = 'x-small'
            y = 0.85

        cmtaxes = []
        for _i, _axcmt in enumerate(axcmts):
            cmtaxes.append(fig.add_subplot(beachgs[_i]))
            cmtaxes[_i].axis('equal')
            cmtaxes[_i].set_xlim(0, 1)
            cmtaxes[_i].set_ylim(0, 1)
            cmtaxes[_i].axis('off')

            # Plot beach ball
            axcmts[_i].axbeach(
                cmtaxes[_i], 0.5, y, width=beachwidth*beachfactor,
                facecolor=cmtcolors[_i],
                clip_on=False, linewidth=1)

        if eventdetails:
            if len(axcmts) == 1:
                plot_label(
                    cmtaxes[0], cmtstring(axcmts[0], labels=True), location=17,
                    dist=0.0, box=False, fontfamily='monospace',
                    fontsize=cmtstringfontsize, color=cmtcolors[0], zorder=100)
            else:
                plot_label(
                    cmtaxes[0], cmtstring(axcmts[0], labels=False), location=17,
                    dist=0.0, box=False, fontfamily='monospace',
                    fontsize=cmtstringfontsize, color=cmtcolors[0], zorder=100)

                plot_label(
                    cmtaxes[0], cmtstring(axcmts[0], labelsonly=True),
                    location=12, dist=0.0, box=False, fontfamily='monospace',
                    fontsize=cmtstringfontsize, color='k', zorder=100
                )

        # ss, ds, rs = newsyntcmt.sdr
        # label = f'{ss[0]:3.0f}/{ds[0]:3.0f}/{rs[1]:4.0f}\n'
        # label += f'{ss[1]:3.0f}/{ds[1]:3.0f}/{rs[1]:4.0f}'

        # cmtax2.text(
        #     1, xy_newsyntbeach[1]-0.2,
        #     label, horizontalalignment='right',
        #     verticalalignment='top', transform=cmtax2.transAxes,
        #     bbox={'facecolor': 'none', 'edgecolor': 'none'},
        #     fontfamily='monospace', fontsize='xx-small', color=newsyntcolor)

    # Getting the CMT that is used for the base info
    # Always start with obsdcmt first os that changes with repect to the can
    # be computed
    if (obsdcmt is None) and (syntcmt is None) and (newsyntcmt is None):
        infocmt = None
    else:
        if obsdcmt is not None:
            infocmt = obsdcmt
        elif syntcmt is not None:
            infocmt = syntcmt
        elif newsyntcmt is not None:
            infocmt = newsyntcmt

    # Plot seismograms
    abc = "abc"
    for _i, _component in enumerate(components):

        # Checking for observed trace
        if obs is not None:
            try:
                obstrace = obs.select(component=_component)[0]

            except Exception as e:
                logger(
                    f"Observed {_component} not available for {network}.{station}")
                print(e)
                obstrace = None
        else:
            obstrace = None

        # Checking for synthetic trace
        if syn is not None:

            try:
                syntrace = syn.select(component=_component)[0]
            except Exception:
                logger(
                    f"Synthetic {_component} not available for {network}.{station}")
                syntrace = None
        else:
            syntrace = None

        # Checking for new synthetic trace
        if newsyn is not None:

            try:
                newsyntrace = newsyn.select(component=_component)[0]
            except Exception:
                logger(
                    f"Synthetic {_component} not available for {network}.{station}")
                newsyntrace = None
        else:
            newsyntrace = None

        # Read custom windows
        if window_dict is not None:
            if obstrace is not None:
                if obstrace.id in window_dict:
                    window_list = window_dict[obstrace.id]
                else:
                    window_list = None
            else:
                window_list = None
        else:
            window_list = None

        # Set some plotting parameters
        legend = True if _i == 0 else False
        labelbottom = True if _i == Ncomp-1 else False

        # Plot the seismograms for compoenent _comp
        plot_seismograms(
            obsd=obstrace, synt=syntrace, syntf=newsyntrace,
            cmtsource=obsdcmt, tag=tag, processfunc=processfunc,
            ax=axes[_i], labelbottom=labelbottom,
            annotations=annotations, labelobsd=labelobsd,
            labelsynt=labelsynt, labelnewsynt=labelnewsynt,
            legend=legend, timescale=timescale, windows=windows,
            window_list=window_list,
            obsdcolor=obsdcolor, syntcolor=syntcolor,
            newsyntcolor=newsyntcolor,
            xlim_in_seconds=xlim_in_seconds)

        # Need some escape for envelope eventually
        axes[_i].set_yticklabels([])
        axes[_i].set_facecolor('none')

        if xlim_in_seconds:
            xmin = xlim_in_seconds[0]/timescale
            xmax = xlim_in_seconds[1]/timescale
        else:
            xmin, xmax = 0, None

        # Set x axis limits
        axes[_i].set_xlim(xmin, xmax)

        # Get and set y axis limits
        _, _, ymin, ymax = get_limits(axes[_i])
        yscale_max = np.max(np.abs([ymin, ymax]))

        if annotations:
            yscale_max *= 1.5

        axes[_i].set_ylim(-yscale_max, yscale_max)

        if _i == Ncomp-1:
            axes[_i].spines.left.set_visible(False)
            axes[_i].spines.right.set_visible(False)
            axes[_i].spines.top.set_visible(False)
            axes[_i].spines.bottom.set_visible(True)

            # This moves the x axis spine away from the annotations!
            outward = 35 if annotations else 5
            axes[_i].spines.bottom.set_position(('outward', outward))
            axes[_i].yaxis.set_ticks([])
            axes[_i].tick_params(axis='x', which='both', bottom=True,
                                 top=False, direction='in')
        else:
            axes[_i].spines.left.set_visible(False)
            axes[_i].spines.right.set_visible(False)
            axes[_i].spines.top.set_visible(False)
            axes[_i].spines.bottom.set_visible(False)
            axes[_i].xaxis.set_ticks([])
            axes[_i].yaxis.set_ticks([])

        # Plot figure label
        if letters:
            plot_label(
                axes[_i], abc[_i] + ")", location=5, box=False,
                fontfamily='monospace')

        if _i == 0:

            if (latitude is not None) and (longitude is not None):
                slat = latitude
                slon = longitude
            elif obstrace is not None:
                if hasattr(obstrace.stats, 'latitude') \
                        and hasattr(obstrace.stats, 'longitude'):
                    slat = obstrace.stats.latitude
                    slon = obstrace.stats.longitude
            else:
                slat = None
                slon = None

            header = cmtheader(
                network=network, station=station, slat=slat, slon=slon,
                cmt=infocmt, periodrange=periodrange)

            # Plot the label
            plot_label(
                axes[0], header, location=6, box=False,
                fontfamily='monospace', dist=0.01, fontsize='large')

    # xlabel
    if timescale == 1:
        unit = f's'
    elif timescale == 60:
        unit = 'min'
    elif timescale == 60*60:
        unit = 'h'
    elif timescale == 60*60*24:
        unit = 'D'
    else:
        unit = f's/{timescale}s'

    axes[-1].set_xlabel(f'Time [{unit}]')

    if map:
        if (latitude is not None) and (longitude is not None):
            slat = latitude
            slon = longitude

            # Create inset with station locations in the center
        elif obstrace is not None:
            if hasattr(obstrace.stats, 'latitude') \
                    and hasattr(obstrace.stats, 'longitude'):
                slat = obstrace.stats.latitude
                slon = obstrace.stats.longitude

        else:
            # raise ValueError(
            #     'If you want to plot a map, you have to either provide the\n'
            #     'latitude and longitude of the station in question,\n'
            #     'or make sure that the Trace.Stats have longitude and '
            #     'latitude.')
            slat = None
            slon = None

        mapax = fig.add_subplot(GSL[0])
        mapax.axis('off')

        # If we have cmtsource center map around
        if obsdcmt is not None:
            clat, clon = obsdcmt.latitude, obsdcmt.longitude
            cmtplot = True
        elif syntcmt is not None:
            clat, clon = syntcmt.latitude, syntcmt.longitude
            cmtplot = True
        elif newsyntcmt is not None:
            clat, clon = newsyntcmt.latitude, newsyntcmt.longitude
            cmtplot = True
        else:
            clat, clon = slat, slon
            cmtplot = False

        if clat is None or clon is None:
            raise ValueError('For some reason I dont have a central (lat/lon)')

        if midpointmap:
            if slat is not None and slon is not None:
                # Get midpoint of the station and the cmt
                mlat, mlon = geomidpointv(clat, clon, slat, slon)
            else:
                # Set midpoint to cmt
                mlat, mlon = clat, clon

            # Projection
            projection = Orthographic(
                central_longitude=mlon, central_latitude=mlat)
        else:
            # Create projection with station or earthquake at center
            projection = AzimuthalEquidistant(
                central_longitude=clon, central_latitude=clat)

        # Create map axes
        mapax = axes_from_axes(
            mapax, 90124, extent=[0.0, 0.0, 0.9, 0.9/get_aspect(mapax)],
            projection=projection)
        mapax.set_global()

        # Plot Map
        if midpointmap:
            plot_map(zorder=-2)
        mapax.coastlines(lw=0.25)
        mapax.gridlines(lw=0.25, ls='-',
                        color=(0.75, 0.75, 0.75), zorder=-1)

        # Plot station
        if slat is not None and slon is not None:
            mapax.plot(
                slon, slat, 'v',
                markerfacecolor=(0.8, 0.2, 0.2),
                markeredgecolor='k', transform=PlateCarree())

        # Plotting Cmtsource location
        if cmtplot:

            mapax.plot(
                clon, clat, '*',
                markerfacecolor=(0.2, 0.2, 0.8),
                markeredgecolor='k', transform=PlateCarree())

        # Plotting Great Circle Path
        if slat is not None and slon is not None \
                and clat is not None and clon is not None:

            mapax.plot(
                [clon, slon], [clat, slat], '-k',
                transform=Geodetic(), zorder=0)

    # Arrow parameters
    arrowprops = dict(lw=0.25, ls='-', ec='k', width=0.01)
    scale = 5.0  # axes length per degree

    if plot_beach and map and ((obsdcmt is not None and syntcmt is not None) or
                               (obsdcmt is not None and newsyntcmt is not None) or
                               (syntcmt is not None and newsyntcmt is not None)):

        # Legend Key length
        keylength = 0.01
        az_arrow(mapax, 0.0, 0.925, scale*keylength,
                 90, fc='k', zorder=10, **arrowprops)
        plot_label(mapax, f'{keylength:.2f}deg', location=6, dist=-0.01, box=False,
                   fontfamily='monospace', fontsize='xx-small', color='k')

        # Get axes location of the event in fractional axes coordinates
        arrowloc = geo2axes(mapax, infocmt.longitude, infocmt.latitude)

    if plot_beach and obsdcmt is not None and syntcmt is not None:

        syntidx = cmtlabels.index('synt')

        if eventdetails:
            compstring = cmtstring(syntcmt, obsdcmt, labels=False)

            plot_label(
                cmtaxes[syntidx], compstring, location=17, dist=0.0, box=False,
                fontfamily='monospace', fontsize=cmtstringfontsize,
                color=cmtcolors[syntidx], zorder=100)

        # Get some geographical data
        dist1, az1, baz1 = gps2dist_azimuth(
            obsdcmt.latitude, obsdcmt.longitude,
            syntcmt.latitude, syntcmt.longitude)
        arrow_length = dist1*360/1000.0/(2*np.pi*6371.0)

        if map:
            if midpointmap:
                # Get fixed arrow directions
                lat2, lon2 = reckon(
                    syntcmt.latitude, syntcmt.longitude, 2, az1)
                p_arrowdir = geo2axes(
                    mapax, lon2, lat2)
                az1 = 90-np.degrees(
                    np.arctan2(
                        p_arrowdir[1]-arrowloc[1],
                        p_arrowdir[0]-arrowloc[0])
                )

            az_arrow(
                mapax, *arrowloc, scale*arrow_length, az1, fc=cmtcolors[syntidx],  zorder=10,
                **arrowprops)

    if plot_beach and (obsdcmt is not None) and (newsyntcmt is not None):

        # Get axes index
        newsyntidx = cmtlabels.index('newsynt')

        if eventdetails:

            #  Get the difference between synthetic and observed
            compstring = cmtstring(newsyntcmt, obsdcmt, labels=False)

            plot_label(
                cmtaxes[newsyntidx], compstring, location=17, dist=0.0,
                box=False, fontfamily='monospace', fontsize=cmtstringfontsize,
                color=cmtcolors[newsyntidx], zorder=100)

        # Get some geographical data
        dist2, az2, baz2 = gps2dist_azimuth(
            obsdcmt.latitude, obsdcmt.longitude,
            newsyntcmt.latitude, newsyntcmt.longitude)
        arrow_length = dist2*360/1000.0/(2*np.pi*6371.0)

        if map:
            if midpointmap:
                # Get fixed arrow directions
                lat2, lon2 = reckon(newsyntcmt.latitude,
                                    newsyntcmt.longitude, 2, az2)
                p_arrowdir = geo2axes(
                    mapax, lon2, lat2)
                az2 = 90-np.degrees(
                    np.arctan2(
                        p_arrowdir[1]-arrowloc[1],
                        p_arrowdir[0]-arrowloc[0])
                )

            az_arrow(
                mapax, *arrowloc, scale*arrow_length, az2, fc=cmtcolors[newsyntidx],
                zorder=10, **arrowprops)

    matplotlib.rcParams.update({'font.family': defff})

    return fig, axes
