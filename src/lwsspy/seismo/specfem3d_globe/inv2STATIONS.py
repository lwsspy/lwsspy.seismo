import os
from obspy import Inventory


def inv2STATIONS(inv: Inventory, outputfilename: str = "./STATIONS"):
    """
    Takes in an obspy inventory and creates a STATIONS file from

    Args:
        inv (Inventory):
            Input inventory
        outputfilename (str, optional):
            Outputfilename. Defaults to "./STATIONS".

    Last modified: Lucas Sawade, 2020.09.24 12.00 (lsawade@princeton.edu)

    """

    # Check whether file directory exists and if it doesn't create it
    dirname = os.path.dirname(outputfilename)
    try:
        os.makedirs(dirname)
    except FileExistsError:
        # directory already exists
        pass

    with open(outputfilename, 'w') as fh:
        for network in inv:
            for station in network:
                # Get station parameters
                lat = station.latitude
                lon = station.longitude
                elev = station.elevation
                burial = 0.0

                # Write line
                fh.write("%-9s %5s %15.4f %12.4f %10.1f %6.1f\n"
                         % (station.code, network.code,
                            lat, lon, elev, burial))
