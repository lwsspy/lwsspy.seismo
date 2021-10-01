import os.path as p
from copy import deepcopy
from typing import Union


from .cmt2STATIONS import cmt2STATIONS



def getsimdirSTATIONS(specfemdir: str,
                      duration: float = 11000.0,
                      startoffset: float = -300.0, endoffset: float = 0.0,
                      network: Union[str, None] = "IU,II,G",
                      station: Union[str, None] = None,
                      client: Union[str, None] = "IRIS"):
    """
    This one takes in a specfemdirectory that contains a cmtsolution and creates
    a corresponding stations file from it.

    Args:
        specfemdir (str):
            specfem3d_globe simulation directory
        duration (float, optional):
            Duration of the recording. Defaults to 11000.0.
        startoffset (float, optional):
            Starttime offset in seconds. Defaults to -300.0.
        endoffset (float, optional):
            Endtime offset in seconds. Defaults to 0.0.
        network (Union[str, None], optional):
            Networks to be requested. Defaults to "IU,II,G".
        station (Union[str, None], optional):
            Stations to be requested. Defaults to None.
        client (Union[str, None], optional):
            Clients to be used. Defaults to "IRIS".
    """

    # Get inputs to cmt2STATIONS
    input_dict = deepcopy(locals())
    input_dict.pop('specfemdir')

    # CMT
    cmtfilename = p.join(specfemdir, "DATA", "CMTSOLUTION")

    # STATIONS file
    outputfilename = p.join(specfemdir, "DATA", "STATIONS")

    # Create STATIONS file from CMT
    cmt2STATIONS(cmtfilename, **input_dict, outputfilename=outputfilename)
