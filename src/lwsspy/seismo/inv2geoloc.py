from typing import Tuple
from obspy import Inventory


def inv2geoloc(inv: Inventory) -> Tuple[list, list]:
    """takes in inventory and returns a tuple of ttwo lists containing the
    latitudes, and longiudes of the stations in the inventory.

    Parameters
    ----------
    inv : Inventory
        Obspy Inventory

    Returns
    -------
    Tuple[list, list]
        [Latitudes, Longitudes]

    """

    latitudes = []
    longitudes = []

    for network in inv:
        for station in network:
            # Get station parameters
            latitudes.append(station.latitude)
            longitudes.append(station.longitude)

    return latitudes, longitudes
