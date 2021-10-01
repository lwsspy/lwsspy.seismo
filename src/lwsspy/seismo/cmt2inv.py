from typing import Union
from obspy.clients.fdsn import Client
# Internal
from .source import CMTSource


def cmt2inv(cmtfilename: str, duration: float = 11000.0,
            startoffset: float = -300.0, endoffset: float = 0.0,
            network: Union[str, None] = "IU,II,G",
            station: Union[str, None] = None,
            client: Union[str, None] = "IRIS"):
    """Takes in a CMTSOLUTION file and creates requests an inventory.

    Args:
        cmtfilename (str):
            CMTSOLUTION file name
        duration (float, optional):
            Duration of recording. Defaults to 11000.0.
        startoffset (float, optional):
            Starttime offset. Defaults to -300.0.
        endoffset (float, optional):
            Endtime offset. Defaults to 0.0.
        network (Union[str, None], optional):
            Networks to request. Defaults to "IU,II,G".
        station (Union[str, None], optional):
            Stations to request. Defaults to None.
        client (Union[str, None], optional):
            Clients to use. Defaults to "IRIS".

    Returns:
        Inventory: Inventory of stations.

    Last modified: Lucas Sawade, 2020.09.24 12.00 (lsawade@princeton.edu)

    """

    # Read in source
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfilename)

    # Set request parameters
    cli = Client(client)
    starttime = cmtsource.origin_time + startoffset
    endtime = cmtsource.origin_time + duration + endoffset

    # Get stations
    inv = cli.get_stations(network=network, station=station,
                           starttime=starttime, endtime=endtime)

    return inv
