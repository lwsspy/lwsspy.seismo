from typing import Tuple
from obspy import Inventory


def inv2net_sta(inv: Inventory) -> Tuple[list, list]:
    """takes in inventory and returns a tuple of ttwo lists containing the
    networks, and stations of the stations in the inventory.

    Parameters
    ----------
    inv : Inventory
        Obspy Inventory

    Returns
    -------
    Tuple[list, list]
        [Networks, Stations]

    """

    networks = []
    stations = []

    for network in inv:
        for station in network:
            # Get station parameters
            networks.append(network.code)
            stations.append(station.code)

    return networks, stations
