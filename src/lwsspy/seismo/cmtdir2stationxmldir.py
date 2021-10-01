# External
import os
import os.path as p
from copy import deepcopy
from glob import glob
from typing import Union

# Internal
from .cmt2stationxml import cmt2stationxml
from ..math.magnitude import magnitude


def cmtdir2stationxmldir(cmtdir: str, stationxmldir: str,
                         duration: float = 11000.0,
                         startoffset: float = -300.0, endoffset: float = 0.0,
                         network: Union[str, None] = "IU,II,G",
                         station: Union[str, None] = None,
                         client: Union[str, None] = "IRIS"):
    """
    Takes in CMTSOLUTION directory and creates a stationxml directory

    Args:
        cmtdir (str):
            Path to cmt directory.
        stationxmldir (str):
            Path to stationxmldir
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

    Last modified: Lucas Sawade, 2020.09.24 15.30 (lsawade@princeton.edu)

    """
    # Parse all inputs to the cmt2inv function except the STATIONS file
    input_dict = deepcopy(locals())
    input_dict.pop('cmtdir')
    input_dict.pop('stationxmldir')

    # Get all cmt files
    cmtfiles = glob(p.join(cmtdir, "*"))

    # Create directory if it doesnt exist
    try:
        os.makedirs(stationxmldir)
    except FileExistsError:
        # directory already exists
        pass

    Nev = len(cmtfiles)
    mag = magnitude(Nev)
    # Loop over files and write
    for _i, _cmtfile in enumerate(cmtfiles):
        outname = p.join(stationxmldir, p.basename(_cmtfile) + ".xml")
        # Print progress
        print(f"#{_i+1:0>{mag+1}}/{Nev}: {_cmtfile} --> {outname}")

        cmt2stationxml(_cmtfile, **input_dict, outputfilename=outname)
