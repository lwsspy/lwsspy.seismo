
#!/bin/env python

import os
import matplotlib.pyplot as plt
import numpy as np
from obspy.imaging.beachball import beach

from .get_stations import get_stations
from ...source import CMTSource
import lwsspy.maps as lmap
from .read_sem_traces import read_sem_traces


def plot_section(specfemdir, unit='d', comp='Z'):

    # Get stations
    stationtable = get_stations(specfemdir)
    latitudes = stationtable['latitude']
    longitudes = stationtable['longitude']
    stations = stationtable['station']
    networks = stationtable['network']

    # Get source
    source = CMTSource.from_CMTSOLUTION_file(os.path.join(specfemdir, "DATA", "CMTSOLUTION")) 

    # Compute epicentral distances in degrees
    dists = lmap.haversine(source.latitude, source.longitude, latitudes, longitudes)

    pos = np.argsort(dists)


    # Read Traces
    tracetable = read_sem_traces(specfemdir, unit=unit)
    Nt = len(tracetable['tr'][0])

    # Create plot array
    trplot = np.zeros((len(stations), Nt))
    addplot = np.tile(np.sort(dists), (Nt,1)).T
    tplot = np.tile(tracetable['t'][0], (len(stations), 1)).T

    # Populate plot array
    for _i, (_sta, _net) in enumerate(zip(stations[pos], networks[pos])):
        row_idx = np.where(
            (tracetable['network'] == _net)
            & (tracetable['station'] == _sta)
            & (tracetable['channel'] == f"BX{comp}")
        )[0]
        
        if len(row_idx) < 1:
            trplot[_i, :] = np.nan
        else:
            trplot[_i, :] = tracetable["tr"][row_idx, :]

    # Plot figure
    plt.figure()
    ax = plt.subplot(111)
    # plt.plot((trplot/np.std(trplot) + addplot).T, 'k')
    # plt.pcolormesh(addplot, tplot.T, trplot/np.max(np.abs(trplot), axis=1)[:, np.newaxis], cmap='gray_r', shading='auto',
    #                vmin=-1.0, vmax=1.0)
    plt.imshow(trplot/np.max(np.abs(trplot), axis=1)[:, np.newaxis], cmap='gray_r', aspect='auto',
                   vmin=-1.0, vmax=1.0)
    ax.invert_yaxis()

    # ax.set_xlim(fextent[:2])
    # ax.set_ylim(fextent[2:])
    # plt.plot(longitudes, latitudes, 'v', markeredgecolor='k', markerfacecolor=(0.8,0.1,0.1))
    
    # ax.add_collection(beach(source.tensor, width=50, xy=(source.longitude, source.latitude), axes=ax))

    plt.show(block=True)


    