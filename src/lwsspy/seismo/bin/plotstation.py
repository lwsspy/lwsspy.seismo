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
from sys import argv, exit
from obspy import read, Inventory
from obspy.geodetics.base import gps2dist_azimuth
from numpy import pi
import matplotlib.pyplot as plt
from lwsspy.seismo.source import CMTSource
from lwsspy.seismo.read_inventory import flex_read_inventory as read_inventory
from lwsspy.seismo.process.process import process_stream
from lwsspy.seismo.plot_seismogram import plot_seismogram_by_station
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
    procdict = dict(
        starttime=cmt.cmt_time,
        endtime=cmt.cmt_time + 100 * 55.0,
        periodmin=periodmin,
        periodmax=periodmax,
        rr=True,
        rs=True,
        rotate=True,
        sampling_rate=1.0
    )

    # Process all data
    pdata = proc(data, inv, cmt, **procdict)

    # Set remove response flag to False for synthetic processing
    procdict['rr'] = False

    # Modify the inventory for synthetics
    minv = synth_inv(inv, mode="mode")
    sinv = synth_inv(inv, mode="sem")

    # Process modes
    pmode = proc(mode, minv, cmt, **procdict)

    # Process SEM synthetics
    psf25 = proc(sf25, sinv, cmt, **procdict)

    # Update plotting options
    updaterc()

    # Plot Seismograms
    plot_seismogram_by_station(
        network, station,
        obsd=pdata, synt=pmode, newsynt=psf25,
        cmtsource=cmt,
        labelobsd="Data",
        labelsynt="Modes",
        labelnewsynt="SEM3D",
    )

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
