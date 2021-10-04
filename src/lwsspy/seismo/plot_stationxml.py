# External
import numpy as np
import matplotlib.pyplot as plt
from obspy import read_inventory
from cartopy.crs import PlateCarree, Mollweide

# Internal
# from .. import inv2geoloc
# from .. import plot_map
# from .. import fix_map_extent
import lwsspy as lpy


def plot_station_xml(filename: str, outputfile: str or None = None):
    """Plots station_xml to map

    Parameters
    ----------
    filename : str
        StationXML

    """

    # Get latitudes and longitudes
    inv = read_inventory(filename)

    # Get latitudes and longitudes
    lat, lon = lpy.seismo.inv2geoloc(inv)

    # Get aspect
    minlat, maxlat = np.min(lat), np.max(lat)
    minlon, maxlon = np.min(lon), np.max(lon)

    # Get extent
    extent = lpy.maps.fix_map_extent([minlon, maxlon, minlat, maxlat])

    aspect = (extent[1] - extent[0])/(extent[3] - extent[2])

    # Plot things
    plt.figure(figsize=(aspect*4, 4))
    ax = plt.axes(projection=Mollweide())
    ax.gridlines()
    lpy.maps.plot_map(projection=Mollweide())
    ax.plot(lon, lat, 'v', label="Stations", markeredgecolor='k',
            markerfacecolor=(0.8, 0.3, 0.3), transform=PlateCarree())
    # ax.set_extent(extent)

    if outputfile is not None:
        plt.savefig(outputfile)
    else:
        plt.show()
        return ax
