
from typing import Callable, Optional, Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from obspy import Trace, Stream
from obspy.geodetics.base import gps2dist_azimuth
from pandas import period_range
from .source import CMTSource
from lwsspy.plot.plot_label import plot_label
from lwsspy.plot.get_limits import get_limits
from lwsspy.math.envelope import envelope
from lwsspy.signal.xcorr import xcorr, correct_window_index
from lwsspy.signal.norm import dnorm2, norm2
from lwsspy.signal.dlna import dlna


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


def get_mcsta(d, s, dt, npts, win, taper=1.0):

    # Get left and right indeces
    l, r = win.left, win.right

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
    ax1.legend(loc='upper right', frameon=False, ncol=3, prop={'size': 11})
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
        processfunc: Callable = lambda x: x,
        annotations: bool = True,
        labelbottom: bool = False,
        labelobsd: str = None,
        labelsynt: str = None,
        labelnewsynt: str = None,
        legend: bool = True,
        windows: bool = True,
        timescale: float = 1.0):

    if ax is None:
        ax = plt.gca()

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

    maxdisp = 0.0
    if plotobsd:
        times_obsd = offset_obsd + obsd.stats.delta * \
            np.arange(obsd.stats.npts)
        pobsd = processfunc(obsd.data)
        ax.plot(np.array(times_obsd)/timescale, pobsd, color="black",
                linewidth=0.75, label=labelobsd)
        obsdmax = np.max(np.abs(pobsd))
        maxdisp = obsdmax if obsdmax > maxdisp else maxdisp

    if plotsynt:
        times_synt = offset_synt + synt.stats.delta * \
            np.arange(synt.stats.npts)
        psynt = processfunc(synt.data)
        ax.plot(np.array(times_synt)/timescale, psynt, color="red",
                linewidth=0.75, label=labelsynt)
        syntmax = np.max(np.abs(psynt))
        maxdisp = syntmax if syntmax > maxdisp else maxdisp

    if plotsyntf:
        times_syntf = offset_syntf + syntf.stats.delta * \
            np.arange(syntf.stats.npts)
        psyntf = processfunc(syntf.data)
        ax.plot(np.array(times_syntf)/timescale, processfunc(syntf.data), color="blue",
                linewidth=0.75, label=labelnewsynt)
        syntfmax = np.max(np.abs(psyntf))
        maxdisp = syntfmax if syntfmax > maxdisp else maxdisp

    if legend:
        ax.legend(loc='lower right', frameon=False, ncol=2,
                  prop=dict(size=11, family='monospace'),
                  bbox_to_anchor=(1., 1.), borderaxespad=0.0)
    ax.tick_params(labelbottom=labelbottom, labeltop=False)

    # Setting top left corner text manually
    # if isinstance(tag, str):
    #     label = f"{trace_id}\n{tag.capitalize()}"
    # else:
    #     label = f"{trace_id}"
    # plot_label(ax, label, location=1, dist=0.005, box=False)

    if plotobsd and (plotsynt or plotsyntf) and (windows):

        # print("trying to print windows")
        try:
            for _i, win in enumerate(obsd.stats.windows):

                # Get window length
                left = times_obsd[win.left]/timescale
                right = times_obsd[win.right]/timescale

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
                        win, taper=taper)

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
                            verticalalignment=va, color='r')

                    if plotsyntf:
                        m, c, s, t, a = get_mcsta(
                            pobsd, psyntf, obsd.stats.delta, obsd.stats.npts,
                            win, taper=taper)

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
                                verticalalignment=va, color='b')

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

    plot_label(ax, "\n\n" + dispstring, location=8, box=False,
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
        plot_label(ax, "\n\n\nM:\n\nC:\n\nS:", location=8, box=False,
                   fontfamily='monospace', dist=0.01, fontsize=fontsize)

        # Create strings
        addstring0 = (
            f"\n\n\n   {FS:.3f}"
            f"\n\n   {CS:.3f}"
            f"\n\n   {SS:.3f}"
        )
        addstring1 = (
            f"\n\n\n\n   {FF:.3f}"
            f"\n\n   {CF:.3f}"
            f"\n\n   {SF:.3f}"
        )

        plot_label(ax, addstring0, location=8, box=False,
                   fontfamily='monospace', fontsize=fontsize,
                   dist=0.01, color='r')

        plot_label(ax, addstring1, location=8, box=False,
                   fontfamily='monospace', fontsize=fontsize,
                   dist=0.01, color='b')

    elif plotobsd and plotsynt and (plotsyntf is False):
        # Misfit and Cross correlation
        FS = np.sum((psynt-pobsd)**2)/np.sum(pobsd**2)
        CS, TS = xcorr(pobsd, psynt)
        SS = np.sum(np.roll(psynt, TS) * pobsd)/np.sum(psynt**2)

        plot_label(ax, "\n\n\nM:\nC:\nS:", location=8, box=False,
                   fontfamily='monospace', dist=0.01, fontsize=fontsize)

        # Create strings
        addstring0 = (
            f"\n\n\n   {FS:.3f}"
            f"\n   {CS:.3f}"
            f"\n   {SS:.3f}"
        )

        plot_label(ax, addstring0, location=8, box=False,
                   fontfamily='monospace', dist=0.01, color='r',
                   fontsize=fontsize)


def plot_seismogram_by_station(
        network: str,
        station: str,
        obsd: Optional[Stream] = None,
        synt: Optional[Stream] = None,
        newsynt: Optional[Stream] = None,
        cmtsource: Optional[CMTSource] = None,
        tag: Optional[str] = None,
        compsystem: str = "ZRT",
        logger: Callable = print,
        processfunc: Callable = lambda x: x,
        annotations: bool = False,
        labelobsd: str = None,
        labelsynt: str = None,
        labelnewsynt: str = None,
        periodrange: list = None,
        windows: bool = True):

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

    figfactor = 1.6 if annotations else 1.3
    hspace = 0.3 if annotations else 0.1
    bottom = 0.2 if annotations else 0.1
    fig = plt.figure(figsize=(10, 1.0 + figfactor*Ncomp))
    plt.subplots_adjust(top=0.85 - 0.05 * (3-Ncomp), hspace=hspace,
                        bottom=bottom + 0.05 * (3-Ncomp))
    # get the first line, there might be more
    axes = []
    abc = "abc"
    for _i, _component in enumerate(components):

        axes.append(plt.subplot(Ncomp*100 + 11 + _i))

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

        if syn is not None:

            try:
                syntrace = syn.select(component=_component)[0]
            except Exception:
                logger(
                    f"Synthetic {_component} not available for {network}.{station}")
                syntrace = None
        else:
            syntrace = None

        if newsyn is not None:

            try:
                newsyntrace = newsyn.select(component=_component)[0]
            except Exception:
                logger(
                    f"Synthetic {_component} not available for {network}.{station}")
                newsyntrace = None
        else:
            newsyntrace = None

        # Set some plotting parameters
        legend = True if _i == 0 else False
        labelbottom = True if _i == Ncomp-1 else False

        # Plot the seismograms for compoenent _comp
        plot_seismograms(obsd=obstrace, synt=syntrace, syntf=newsyntrace,
                         cmtsource=cmtsource, tag=tag, processfunc=processfunc,
                         ax=axes[_i], labelbottom=labelbottom,
                         annotations=annotations, labelobsd=labelobsd,
                         labelsynt=labelsynt, labelnewsynt=labelnewsynt,
                         legend=legend, timescale=3600.0, windows=windows)

        # Get limits for seismograms
        # Need some escape for envelope eventually
        axes[_i].set_yticklabels([])
        axes[_i].set_facecolor('none')

        xmin, xmax, ymin, ymax = get_limits(axes[_i])
        yscale_max = np.max(np.abs([ymin, ymax]))

        if annotations:
            yscale_max *= 1.5

        axes[_i].set_xlim(xmin, xmax)
        axes[_i].set_ylim(-yscale_max, yscale_max)

        if _i == Ncomp-1:
            axes[_i].spines.left.set_visible(False)
            axes[_i].spines.right.set_visible(False)
            axes[_i].spines.top.set_visible(False)
            axes[_i].spines.bottom.set_visible(True)
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
        plot_label(
            axes[_i], abc[_i] + ")", location=5, box=False,
            fontfamily='monospace')

        if _i == 0:

            # Get filter
            if periodrange is None:
                periodstr = ""
            else:
                periodstr = f", P: {int(periodrange[0]):d}-{int(periodrange[1]):d}s"

            # Get some geographical data
            dist, az, baz = gps2dist_azimuth(
                cmtsource.latitude, cmtsource.longitude,
                obstrace.stats.latitude, obstrace.stats.longitude)
            m2deg = 360/(2*np.pi*6371000.0)

            # Create String
            station_string = (
                f"{cmtsource.cmt_time.strftime('%Y/%m/%d %H:%M:%S')}, "
                f"$\\theta$={cmtsource.latitude:6.2f}, "
                f"$\\phi$={cmtsource.longitude:7.2f}, "
                f"$h$={cmtsource.depth_in_m/1000.0:5.1f}\n"
                f"{station}-{network}   "
                f"$\\Delta$={m2deg*dist:6.2f}, "
                f"$\\alpha$={az:6.2f}, "
                f"$\\beta$={baz:6.2f}"
                f"{periodstr}"
            )

            # Plot the label
            plot_label(
                axes[0], station_string, location=6, box=False,
                fontfamily='monospace', dist=0.0)

    axes[-1].set_xlabel("Time [h]")

    matplotlib.rcParams.update({'font.family': defff})

    return fig, axes
