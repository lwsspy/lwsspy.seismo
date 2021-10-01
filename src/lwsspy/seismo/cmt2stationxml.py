from copy import deepcopy
from typing import Union
from .cmt2inv import cmt2inv
from .inv2stationxml import inv2stationxml


def cmt2stationxml(cmtfilename: str, duration: float = 11000.0,
                   startoffset: float = -300.0, endoffset: float = 0.0,
                   network: Union[str, None] = "IU,II,G",
                   station: Union[str, None] = None,
                   client: Union[str, None] = "IRIS",
                   outputfilename: str = "station.xml"):
    """
    Takes in CMTSOLUTION and writes out StationXML containing all stations
    that contain

    Args:
        cmtfilename (str):
            Path to CMTSOLUTION file.
        duration (float, optional):
            Duration. Defaults to 11000.0.
        startoffset (float, optional):
            Starttime in seconds. Defaults to -300.0.
        endoffset (float, optional):
            Endtime offset in seconds. Defaults to 0.0.
        network (Union[str, None], optional):
            Networks to be requested. Defaults to "IU,II,G".
        station (Union[str, None], optional):
            Stations to be requested. Defaults to None.
        client (Union[str, None], optional):
            Clients to be used. Defaults to "IRIS".
        outputfilename (str, optional):
            Output filename for the StationXML. Defaults to "station.xml".

    Last modified: Lucas Sawade, 2020.09.24 14.00 (lsawade@princeton.edu)

    """

    # Parse all inputs to the cmt2inv function except the STATIONS file
    input_dict = deepcopy(locals())
    input_dict.pop('outputfilename')

    # Get inventory
    inv = cmt2inv(**input_dict)

    # Write station xml
    inv2stationxml(inv, outputfilename=outputfilename)
