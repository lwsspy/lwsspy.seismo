from copy import deepcopy
from typing import Union
from obspy.clients.fdsn import Client

# Internal
from ..cmt2inv import cmt2inv
from .inv2STATIONS import inv2STATIONS


def cmt2STATIONS(cmtfilename: str, duration: float = 11000.0,
                 startoffset: float = -300.0, endoffset: float = 0.0,
                 network: Union[str, None] = "IU,II,G",
                 station: Union[str, None] = None,
                 client: Union[str, None] = "IRIS",
                 outputfilename: str = "STATIONS",):
    """

    Takes in CMTSOLUTION and requests a set of STATIONS through obspy that have
    been recording at the time

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
        outputfilename (str):
            Path to file that is supposed to be written

    Returns:
        None

    Last modified: Lucas Sawade, 2020.09.24 12.00 (lsawade@princeton.edu)

    """
    # Parse all inputs to the cmt2inv function except the STATIONS file
    input_dict = deepcopy(locals())
    input_dict.pop('outputfilename')

    # Get 
    inv = cmt2inv(**input_dict)

    # Write inventory to STATIONS file
    inv2STATIONS(inv, outputfilename=outputfilename)

