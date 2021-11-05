#!/usr/bin/python

import cartopy
import numpy as np
import matplotlib.pyplot as plt
from ... import base as lbase
from ... import math as lmat
from ... import plot as lplt
from ... import maps as lmap


def plot_csv_depth_slice(infile, outfile, label):
    """Takes in a depthslice CSV file and creates a map from it.

    Parameters
    ----------
    infile : str
        CSV file that contains the depth slice
    outfile : str
        name to save the figure to. Exports PDFs only and export a name 
        depending on depth of the slice.
    label : str
        label of the colorbar

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.09.17 19.00

    """

    # Define title:
    if label == 'rho':
        cbartitle = r"$\rho$"
    if label == 'vpv':
        cbartitle = r"$v_{P_v}$"
    if label == 'vsv':
        cbartitle = r"$v_{S_v}$"

    # Load data from file
    data = np.genfromtxt(infile, delimiter=',')[1:, :]

    # Get Depth of slice
    depth = lbase.EARTH_RADIUS_KM - \
        (lbase.EARTH_RADIUS_KM * np.max(data[:, -2]))

    # Convert to geographical
    rho, lat, lon = lmat.cart2geo(data[:, 1], data[:, 2], data[:, 3])

    # Create interpolation Grid
    llon, llat = np.meshgrid(np.linspace(-180.0, 180.1, 1441),
                             np.linspace(-90.0, 90.1, 721))
    # Interpoalte data
    SNN = lmat.SphericalNN(lat, lon)
    interpolator = SNN.interpolator(llat, llon, no_weighting=True)
    datainterp = interpolator(data[:, -1])

    # Create Figure
    lplt.updaterc()
    plt.figure(figsize=(9, 4))
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    ax.set_rasterization_zorder(-10)
    lmap.plot_map(fill=False, zorder=1)

    pmesh = plt.pcolormesh(llon, llat, datainterp,
                           transform=cartopy.crs.PlateCarree(), zorder=-11)
    lplt.plot_label(ax, f"{depth:.1f} km", aspect=2.0,
                        location=1, dist=0.025, box=True)

    c = plt.colorbar(pmesh, fraction=0.05, pad=0.075)
    c.set_label(cbartitle, rotation=0, labelpad=10)
    # plt.show()
    plt.savefig(f"{outfile}_{int(depth):d}km.pdf", dpi=300)


def bin():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='infile',
                        help='CSV file containing the slice info',
                        required=True, type=str)
    parser.add_argument('-o', dest='outfile',
                        help='Output filename to save the pdf plot to',
                        required=True, type=str)
    parser.add_argument('-l', dest='label',
                        help='label: rho, vpv, ...',
                        required=True, type=str)
    args = parser.parse_args()

    # Run
    plot_csv_depth_slice(args.infile, args.outfile, args.label)
