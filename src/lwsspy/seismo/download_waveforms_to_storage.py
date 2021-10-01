import argparse
import os
from typing import Union, List
from obspy import Inventory
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import RectangularDomain, \
    Restrictions, MassDownloader


def download_waveforms_to_storage(
        datastorage: str,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        minimum_length: float = 0.9,
        reject_channels_with_gaps: bool = True,
        network: Union[str, None] = "IU,II,G",
        station: Union[str, None] = None,
        channel: Union[str, None] = "BH*",
        location: Union[str, None] = "00",
        providers: Union[List[str], None] = ["IRIS"],
        minlatitude: float = -90.0,
        maxlatitude: float = 90.0,
        minlongitude: float = -180.0,
        maxlongitude: float = 180.0,
        limit_stations_to_inventory: Union[Inventory, None] = None):

    domain = RectangularDomain(minlatitude=minlatitude,
                               maxlatitude=maxlatitude,
                               minlongitude=minlongitude,
                               maxlongitude=maxlongitude)

    restrictions = Restrictions(
        starttime=starttime,
        endtime=endtime,
        reject_channels_with_gaps=True,
        # Trace needs to be almost full length
        minimum_length=minimum_length,
        network=network,
        channel=channel,
        location=location,
        limit_stations_to_inventory=limit_stations_to_inventory)

    # Datastorage:
    waveform_storage = os.path.join(datastorage, 'waveforms')
    station_storage = os.path.join(datastorage, 'stations')

    # Create massdownloader
    mdl = MassDownloader(providers=providers)
    print(f"\n")
    print(f"{' Downloading data to: ':*^72}")
    print(f"MSEEDs: {waveform_storage}")
    print(f"XMLs:   {station_storage}")

    mdl.download(domain, restrictions, mseed_storage=waveform_storage,
                 stationxml_storage=station_storage)
    print("\n")
    print(72 * "*")
    print("\n")


def bin():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='datastorage',
                        help='Where to save the waveforms and station files',
                        required=True, type=str)
    parser.add_argument('-s', '--starttime', dest='starttime',
                        help='s',
                        required=True, type=str)
    parser.add_argument('-e', '--endtime', dest='endtime',
                        help='endtime',
                        required=True, type=str)
    parser.add_argument('-N', '--networks', dest='networks',
                        help='list of networks, e.g. "IU,II,G" ',
                        default="IU,II,G", required=False, type=str or None)
    parser.add_argument('-S', '--stations', dest='stations',
                        help='list of stations, e.g. "BFO,IBFO" ',
                        default=None, required=False, type=str or None)
    parser.add_argument('-L', '--locations', dest='locations',
                        help='list of locations, e.g. "00,01" ', default="00",
                        required=False, type=str or None)
    parser.add_argument('-C', '--channels', dest='channels',
                        help='list of locations, e.g. "BH*,HHZ,HHN" ',
                        default="BH*", required=False, type=str or None)
    parser.add_argument('-P', '--providers', dest='providers',
                        help='list of providers, e.g. "IRIS ORFEUS" ',
                        default=["IRIS"], required=False, type=list or None)

    args = parser.parse_args()

    download_waveforms_to_storage(
        args.datastorage,
        UTCDateTime(args.starttime),
        UTCDateTime(args.endtime),
        network=args.networks,
        station=args.stations,
        location=args.locations,
        channel=args.channels,
        providers=args.providers)
