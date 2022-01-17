#!/usr/bin/env python
'''
Small script to download event data.

getdata CMTCODE, e.g. getdata C202106062316A
getdata CMTCODE NET STA, e.g. getdata C201606261117A II ARU
getdata CMTCODE NET STA CH, e.g. getdata C201606261117A II ARU BH

Downloads 200 min from the preferred channel. You can specify the network,
station, and channel. The data is downloaded to

$HOME/SEISDATA/OBSERVED/<EVENT>/
    |-- waveforms/
    |-- stations/

And is not processed.

:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2021.01.17 11.45

'''

from os import environ
from os.path import join
from sys import argv, exit
from ..source import CMTSource
from ..download_waveforms_to_storage import download_waveforms_to_storage


def bin():

    network = None
    station = None
    channel = None

    if len(argv) == 1:
        print("\nA CMT file is required for data download.\n\n",
              __doc__)
        exit()

    elif len(argv) == 2:
        cmt = argv[1]

    elif len(argv) == 3:
        cmt = argv[1]
        network = argv[2]

    elif len(argv) == 4:
        cmt = argv[1]
        network = argv[2]
        station = argv[3]

    elif len(argv) == 5:
        cmt = argv[1]
        network = argv[2]
        station = argv[3]
        channel = argv[4] + "*"

    # get startime and endtime from cmtsolution
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmt)
    starttime = cmtsource.origin_time
    endtime = cmtsource.origin_time + 200.0 * 60.0 - 120.0

    # Specify location
    if not cmtsource.eventname[0].isnumeric():
        eventname = cmtsource.eventname
    else:
        eventname = "C" + cmtsource.eventname

    datastorage = join(
        environ["HOME"], "SEISDATA", "OBSERVED", eventname)

    print("Download data for:")
    print(f"{eventname} at {network}.{station}..{channel}")

    # Download data
    download_waveforms_to_storage(
        datastorage=datastorage,
        starttime=starttime,
        endtime=endtime,
        network=network,
        station=station,
        location=None,
        channel=channel,
        providers=["IRIS"])


if __name__ == "__main__":
    bin()
