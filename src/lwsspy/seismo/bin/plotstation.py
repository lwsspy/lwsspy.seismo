#!/usr/bin/env python

"""

Usage:

    plotstation <EVENT> <NET> <STA>

    eg. plotstation C201801230931A II ARU

Small script that checks the $SEISDATA directory for seismograms and station
data and plots the displacement seismograms for a single station. At the moment
only observed, modes, and M25 data can be used and if one is missing that's not
good.


:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2021.01.17 11.45

"""

from os import path, environ
from re import M
from sys import argv, exit
from matplotlib import rcParams
from obspy import read, Inventory
from obspy.geodetics.base import gps2dist_azimuth
from numpy import pi
import matplotlib.pyplot as plt
from lwsspy.seismo.source import CMTSource
from lwsspy.seismo.read_inventory import flex_read_inventory as read_inventory
from lwsspy.seismo.process.process import process_stream
from lwsspy.seismo.plot_seismogram import plot_seismogram_by_station
from lwsspy.seismo import PROCD
from lwsspy.plot.updaterc import updaterc


def printinv(inv):
    # Modify to match the synthetic metadata
    for network in inv:
        for station in network:
            for channel in station:
                print(channel)


def synth_inv(inv: Inventory, mode='mode'):

    # Copy inventory
    sinv = inv.copy()

    if mode == "mode":
        location = "S1"
        channelext = "LX"
    elif mode == "sem":
        location = "S3"
        channelext = "MX"

    # Dip and Azimuth
    orientation = {"Z": (90, 0),
                   "N": (0, 0),
                   "E": (0, 90)}

    # Modify to match the synthetic metadata
    for network in sinv:
        for station in network:
            for channel in station:

                if channel.code[-1] in ['1', 'N']:
                    channel.dip, channel.azimuth = orientation["N"]
                    channel.code = channelext + 'N'
                    channel.location_code = location

                elif channel.code[-1] in ['2', 'E']:
                    channel.dip, channel.azimuth = orientation["E"]
                    channel.code = channelext + 'E'
                    channel.location_code = location

                elif channel.code[-1] in ['Z']:
                    channel.dip, channel.azimuth = orientation["Z"]
                    channel.code = channelext + 'Z'
                    channel.location_code = location
                else:
                    raise ValueError(f"{channel.code} not implemented...")
    return sinv


def proc(
        st, inv, event, starttime, endtime, periodmin, periodmax, rr=False,
        rs=False, rotate=False, sampling_rate=1.0):

    pst = st.copy()
    pst.detrend('linear')
    pst.detrend('demean')
    pst.taper(max_percentage=0.025, type='cosine')
    pst.filter('bandpass', freqmin=1/periodmax,
               freqmax=1/periodmin, zerophase=True)

    if rr:
        pst.remove_response(inv, water_level=100.0,
                            output="DISP", zero_mean=False)
        pst.filter('bandpass', freqmin=1/periodmax,
                   freqmax=1/periodmin, zerophase=True)

    if rotate:

        # Attach coordinates and back_azimuth to traces
        for tr in pst:

            # Get station coordinates
            coord_dict = inv.get_coordinates(tr.get_id())

            lat = coord_dict["latitude"]
            lon = coord_dict["longitude"]

            # Add location to trace
            tr.stats.latitude = lat
            tr.stats.longitude = lon

            # Get distance to earthquake
            m2deg = 2*pi*6371000.0/360
            tr.stats.distance, tr.stats.azimuth, tr.stats.back_azimuth = gps2dist_azimuth(
                event.latitude, event.longitude, lat, lon)
            tr.stats.distance /= m2deg

        pst.rotate("->ZNE", inventory=inv)
        pst.rotate("NE->RT", inventory=inv)

    if rs:
        npts = int((endtime - starttime) * sampling_rate) + 1
        tr.interpolate(1.0, starttime=starttime, npts=npts)
    else:
        pst.trim(starttime, endtime)

    return pst


def bin():

    # Check if enough arguments
    if len(argv) < 4:
        print(__doc__)
        exit()
    elif len(argv) > 4 and len(argv) < 6:
        print(__doc__)
        exit()
    elif len(argv) == 6:
        event = argv[1]
        network = argv[2]
        station = argv[3]
        periodmin = float(argv[4])
        periodmax = float(argv[5])
    else:
        event = argv[1]
        network = argv[2]
        station = argv[3]
        periodmin = 150
        periodmax = 400

    plotstation(event, network, station, periodmin, periodmax)


def plotstation(event, network, station, periodmin, periodmax):

    # Directories
    GCMT = environ["GCMT"]
    DATA = environ["SOBSERVED"]
    MODES = path.join(environ["SMDATA"], "MODES")
    M25 = environ["M25DATA"]

    # File locations
    eventfile = path.join(
        GCMT, event)
    stationfile = path.join(
        DATA, event, 'stations', f"{network}.{station}.xml")
    wavefiles = path.join(
        DATA, event, 'waveforms', f"{network}.{station}.*.LH*.mseed")
    modefiles = path.join(
        MODES, event, f"{network}.{station}.LX*.modes.sac")
    semfiles = path.join(
        M25, event, f"{network}.{station}.MX*.sem.sac")

    # Get event
    cmt = CMTSource.from_CMTSOLUTION_file(eventfile)

    # Get station info
    inv = read_inventory(stationfile)

    # Get data
    data = read(wavefiles)

    # Get modes
    mode = read(modefiles)

    # Get M25 data
    sf25 = read(semfiles)

    # Observed processing parameters:
    fmin, fmax = 1/periodmax, 1/periodmin
    fminn = fmin - 0.01*fmin
    fmaxx = fmax + 0.01*fmax
    procdict = dict(
        starttime=cmt.cmt_time,
        endtime=cmt.cmt_time + 3 * 3600.0,
        pre_filt=[fminn, fmin, fmax, fmaxx],
        remove_response_flag=True,
        rotate_flag=True,
        resample_flag=True,
        sampling_rate=1.0,
        sanity_check=True,
        event_latitude=cmt.latitude,
        event_longitude=cmt.longitude
    )

    PROCD.pop("relative_starttime")
    PROCD.pop("relative_endtime")
    PROCD.update(procdict)

    # Process all data
    pdata = process_stream(data, inv, **PROCD, geodata=True)

    # Set remove response flag to False for synthetic processing
    PROCD['remove_response_flag'] = False
    # False

    # # Modify the inventory for synthetics
    # minv = synth_inv(inv, mode="mode")
    # sinv = synth_inv(inv, mode="sem")

    # Process modes
    pmode = process_stream(mode, inv, **PROCD)

    # Process SEM synthetics
    psf25 = process_stream(sf25, inv, **PROCD)

    # Update plotting options
    updaterc()
    rcParams['font.family'] = 'monospace'

    # Plot Seismograms
    compsystem = "ZRT"
    fig, axes = plot_seismogram_by_station(
        network, station,
        obsd=pdata, synt=pmode, newsynt=psf25,
        cmtsource=cmt,
        labelobsd="Data",
        labelsynt="Modes",
        labelnewsynt="SEM3D",
        compsystem=compsystem,
        periodrange=[periodmin, periodmax]
    )

    rect = plt.Rectangle((1.75, 0), width=0.75,
                         height=len(compsystem) +
                         fig.subplotpars.hspace*(len(compsystem)-1),
                         transform=axes[-1].get_xaxis_transform(),
                         clip_on=False, edgecolor="none", facecolor="gray",
                         linewidth=3, alpha=0.2, zorder=-1)
    axes[-1].add_patch(rect)

    plt.show(block=True)


if __name__ == "__main__":
    bin()


#  def plot_netsta(network, station):
#    ...:     data = read(f"testdata/waveforms/{network}.{station}.00.LH*")
#    ...:     modes = read(f'Downloads/C201801230931A.1D.sac/{network}.{station}.LX*')
#    ...:     sem = read(f'Downloads/C201801230931A.3D.sac/{network}.{station}.MX*')
#    ...:     pdata = process_stream(data.copy(), inv, **pdict, event_latitude=56.1200, event_longitude=-149.1800, remove_response_flag=Tru
#    ...: e)
#    ...:     psem = process_stream(sem.copy(), inv, **pdict, event_latitude=56.1200, event_longitude=-149.1800, remove_response_flag=False
#    ...: )
#    ...:     pmodes = process_stream(modes.copy(), inv, **pdict, event_latitude=56.1200, event_longitude=-149.1800)
#    ...:     plot_seismogram_by_station(network, station, obsd=pdata, synt=pmodes, newsynt=psem, labelobsd='Obs', labelsynt='Modes', label
#    ...: newsynt='SEM')
#    ...:     plt.savefig(f'{network}.{station}.pdf')
#    ...:     plt.close()
