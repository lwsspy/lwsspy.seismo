from obspy import Inventory


def inv2stationxml(inv: Inventory, outputfilename: str = "station.xml"):
    """
    Tankes in inv and writes stationxml.

    Args:
        inv (Inventory):
            Obspy station inventory
        outputfilename (str, optional):
            Outputfilename. Defaults to "station.xml".

    Last modified: Lucas Sawade, 2020.09.24 13.00 (lsawade@princeton.edu)

    """

    # write inventory to file
    inv.write(outputfilename, format='STATIONXML')
