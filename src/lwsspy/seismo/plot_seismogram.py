
from typing import Callable, Optional, Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from obspy import Trace, Stream
from .source import CMTSource
from ..plot.plot_label import plot_label
from ..plot.get_limits import get_limits
from ..math.envelope import envelope


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
        labelnewsynt: str = None):

    if ax is None:
        ax = plt.gca()

    if isinstance(obsd, Trace):
        station = obsd.stats.station
        network = obsd.stats.network
        channel = obsd.stats.channel
        location = obsd.stats.location
    elif isinstance(synt, Trace):
        station = synt.stats.station
        network = synt.stats.network
        channel = synt.stats.channel
        location = synt.stats.location
    elif isinstance(syntf, Trace):
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
        if isinstance(obsd, Trace):
            offset_obsd = obsd.stats.starttime - cmtsource.cmt_time
        if isinstance(synt, Trace):
            offset_synt = synt.stats.starttime - cmtsource.cmt_time
        if isinstance(syntf, Trace):
            offset_syntf = syntf.stats.starttime - cmtsource.cmt_time

    if isinstance(obsd, Trace):
        times_obsd = [offset_obsd + obsd.stats.delta *
                      i for i in range(obsd.stats.npts)]
        ax.plot(times_obsd, processfunc(obsd.data), color="black",
                linewidth=0.75, label=labelobsd)
    if isinstance(synt, Trace):
        times_synt = [offset_synt + synt.stats.delta * i
                      for i in range(synt.stats.npts)]
        ax.plot(times_synt, processfunc(synt.data), color="red",
                linewidth=0.75, label=labelsynt)
    if isinstance(syntf, Trace):
        times_syntf = [offset_syntf + syntf.stats.delta * i
                       for i in range(syntf.stats.npts)]
        ax.plot(times_syntf, processfunc(syntf.data), color="blue",
                linewidth=0.75, label=labelnewsynt)

    ax.legend(loc='upper right', frameon=False, ncol=3, prop={'size': 11})
    ax.tick_params(labelbottom=labelbottom, labeltop=False)

    # Setting top left corner text manually
    if isinstance(tag, str):
        label = f"{trace_id}\n{tag.capitalize()}"
    else:
        label = f"{trace_id}"
    plot_label(ax, label, location=1, dist=0.005, box=False)

    if isinstance(obsd, Trace) and isinstance(synt, Trace):
        try:
            scaleabsmax = 2 * np.max(np.abs(obsd.data))
            for _i, win in enumerate(obsd.stats.windows):
                left = times_obsd[win.left]
                right = times_obsd[win.right]
                re1 = Rectangle((left, -scaleabsmax),
                                right - left, 2*scaleabsmax,
                                color="blue", alpha=0.10, zorder=-1)
                ax.add_patch(re1)

                if annotations:
                    trans = transforms.blended_transform_factory(
                        ax.transData, ax.transAxes)

                    string = \
                        f"dlnA = {win.dlnA:6.2f}\n" \
                        f" mCC = {win.max_cc_value:6.2f}\n" \
                        f"  CC = {win.cc_shift*win.dt:6.2f}"

                    va = 'top' if (_i % 2) == 0 else 'bottom'
                    y = 0.98 if (_i % 2) == 0 else 0.02

                    ax.text(left, y, string, transform=trans,
                            fontdict=dict(size="x-small", family='monospace'),
                            horizontalalignment='left',
                            verticalalignment=va)

        except Exception as e:
            print(e)


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
        labelnewsynt: str = None):

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

    fig = plt.figure(figsize=(12, 8))
    # get the first line, there might be more
    axes = []
    for _i, _component in enumerate(components):

        axes.append(plt.subplot(311 + _i))

        if obs is not None:
            try:
                obstrace = obs.select(component=_component)[0]
            except Exception:
                logger(
                    f"Observed {_component} not available for {network}.{station}")
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

        labelbottom = True if _i == 2 else False
        plot_seismograms(obsd=obstrace, synt=syntrace, syntf=newsyntrace,
                         cmtsource=cmtsource, tag=tag, processfunc=processfunc,
                         ax=axes[_i], labelbottom=labelbottom,
                         annotations=annotations, labelobsd=labelobsd,
                         labelsynt=labelsynt, labelnewsynt=labelnewsynt)

        # Get limits for seismograms
        # Need some escape for envelope eventually
        xmin, xmax, ymin, ymax = get_limits(axes[_i])
        yscale_max = 1.25 * np.max(np.abs([ymin, ymax]))
        axes[_i].set_xlim(xmin, xmax)
        axes[_i].set_ylim(-yscale_max, yscale_max)

    axes[-1].set_xlabel("Time [s]")
