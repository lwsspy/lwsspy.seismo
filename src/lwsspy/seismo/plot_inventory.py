from obspy import Inventory
from cartopy.crs import PlateCarree
import matplotlib.pyplot as plt
import numpy as np
from .. import plot as lplt
from .. import utils as lutil
from .inv2geoloc import inv2geoloc


def plot_inventory(inv: Inventory, *args, ax: plt.Axes = None,
                   cmap: str = 'tab10', mapproj: bool = True, **kwargs):

    if ax is None:
        ax = plt.gca()

    if mapproj:
        transform = PlateCarree()
    else:
        transform = None

    # Get networks
    networks = inv.get_contents()['networks']
    num_net = len(networks)

    # Update the colorcycler to automate the different colors of the stations
    # And networks
    # net_colors = lplt.pick_colors_from_cmap(num_net, cmap)
    colormap = plt.get_cmap(cmap)
    net_colors = [colormap(i) for i in range(num_net)[::-1]]

    for network, col in zip(networks, net_colors):

        # Get subnetwork
        subinv = inv.select(network=network)

        # Get locations
        lat, lon = inv2geoloc(subinv)
        lat, lon = lutil.get_unique_lists(lat, lon)

        # Plot with label
        ax.plot(lon, lat, 'v', *args, markeredgecolor='k', markeredgewidth=0.25,
                markerfacecolor=col, transform=transform,
                label=network, **kwargs)
