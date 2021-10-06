from typing import Tuple
from obspy import Inventory
from .. import maps as lmap
from .inv2geoloc import inv2geoloc


def get_inv_aspect_extent(inv: Inventory) -> Tuple[float, list]:
    """Takes in an Inventory of nettworks and stations, determines the extent
    of geolocations, adds a buffer to the extent and outputs the final
    aspect ratio and extent. If station+buffer extent exceeds geographical 
    limits, the extent is sett to the limits and aspect is set to 2.

    Parameters
    ----------
    inv : Inventory
        Obspy inventory of networks and stations

    Returns
    -------
    Tuple[float, list]
        aspect ratio, extent

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.01.13 14.30

    """

    # Get all lat,lon s
    lat, lon = inv2geoloc(inv)

    # Get aspect
    minlat, maxlat = np.min(lat), np.max(lat)
    minlon, maxlon = np.min(lon), np.max(lon)

    # Get extent
    extent = lmap.fix_map_extent([minlon, maxlon, minlat, maxlat])

    aspect = (extent[1] - extent[0])/(extent[3] - extent[2])

    return aspect, extent
