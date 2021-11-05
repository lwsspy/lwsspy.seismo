from ..read_inventory import flex_read_inventory as read_inventory
from .inv2STATIONS import inv2STATIONS


def stationxml2STATIONS(xmlfilename: str, stationsfilename: str):
    """
    Takes in StationsXML file and output STATIONS file.

    Args:
        xmlfilename (str): stationsXML
        stationsfilename (str): stations

    Last modified: Lucas Sawade, 2020.09.24 13.00 (lsawade@princeton.edu)

    """

    # Get Inventory
    inv = read_inventory(xmlfilename, format="STATIONXML")

    # Write STATIONS file
    inv2STATIONS(inv, outputfilename=stationsfilename)