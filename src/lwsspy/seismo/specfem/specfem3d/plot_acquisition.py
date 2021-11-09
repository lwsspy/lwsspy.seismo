#!/bin/env python

import os
import matplotlib.pyplot as plt
import numpy as np
from obspy.imaging.beachball import beach

from .get_stations import get_stations
from ...source import CMTSource
import lwsspy.maps as lmap

def plot_acquisition(specfemdir, unit='d'):

    # Get stations
    stationtable = get_stations(specfemdir)
    
    # Get source
    source = CMTSource.from_CMTSOLUTION_file(os.path.join(specfemdir, "DATA", "CMTSOLUTION"))

    # 
    latitudes = stationtable['latitude']
    longitudes = stationtable['longitude']

    # Set extent
    extent = [
        np.min(np.append(longitudes, source.longitude)),
        np.max(np.append(longitudes, source.longitude)),
        np.min(np.append(latitudes, source.latitude)),
        np.max(np.append(latitudes, source.latitude))
    ]
    fextent = lmap.fix_map_extent(extent, fraction=0.05)
    


    # Plot figure
    plt.figure()
    ax = plt.subplot(111)
    ax.set_xlim(fextent[:2])
    ax.set_ylim(fextent[2:])
    plt.plot(longitudes, latitudes, 'v', markeredgecolor='k', markerfacecolor=(0.8,0.1,0.1))
    
    ax.add_collection(beach(source.tensor, width=50, xy=(source.longitude, source.latitude), axes=ax))

    plt.show()


    
    
    
    
    
def plot_section(specfemdir):
    pass