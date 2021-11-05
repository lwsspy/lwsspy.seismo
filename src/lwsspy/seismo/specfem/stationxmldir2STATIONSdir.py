# External
import os
import os.path as p
from glob import glob

# Internal
from .stationxml2STATIONS import stationxml2STATIONS
from ...math.magnitude import magnitude


def stationxmldir2STATIONSdir(stationxmldir: str, stationsdir: str):
    """
    Takes in the path to a stationxml directory and outputs the corresponding
    station files in the stationsdir.

    Args:
        stationxmldir (str):
            Path to StationXML directory
        stationsdir (str):
            Path to STATIONS directory
    """

    # Glob the files
    stationxmlfiles = glob(p.join(stationxmldir, "*"))

    Nev = len(stationxmlfiles)
    mag = magnitude(Nev)
    # Create path to stations directory
    for _i, _xmlfile in enumerate(stationxmlfiles):
        outname = p.join(stationsdir,
                         p.basename(_xmlfile).split(".")[0] + ".stations")
        # Print progress
        print(f"#{_i+1:0>{mag+1}}/{Nev}: {_xmlfile} --> {outname}")
        stationxml2STATIONS(_xmlfile, outname)
