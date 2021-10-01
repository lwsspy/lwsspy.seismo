import matplotlib.pyplot as plt
import cartopy
import numpy as np
import lwsspy as lpy


def plot_specfem_xsec_depth(infile, outfile=None, ax=None, cax=None,
                            depth=None):
    """Takes in a Specfem depthslice and creates a map from it.

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
        2020.12.23 16.30

    """

    # Load data from file
    llon, llat, rad, val, _, _, _ = lpy.read_specfem_xsec_depth(
        infile, res=0.25, no_weighting=False)
    extent = [np.min(llon), np.max(llon), np.min(llat), np.max(llat)]

    # Get Depth of slice
    if depth is None:
        depth = lpy.EARTH_RADIUS_KM - np.mean(rad)

    # Create Figure
    lpy.updaterc()

    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.axes(projection=cartopy.crs.Mollweide())

    ax.set_rasterization_zorder(-10)
    lpy.plot_map(fill=False, zorder=1)

    im = ax.imshow(val[::-1, :], extent=extent,
                   transform=cartopy.crs.PlateCarree(), zorder=-15,
                   cmap='rainbow_r', alpha=0.9)

    lpy.plot_label(ax, f"{depth:.1f} km", aspect=2.0,
                   location=2, dist=0.0, box=False)
    c = lpy.nice_colorbar(im, fraction=0.05, pad=0.075,
                          orientation="horizontal", cax=cax)
    # plt.show()

    if outfile is not None:
        plt.savefig(f"{outfile}_{int(depth):d}km.pdf", dpi=300)
    else:
        return ax, c


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
    plot_specfem_xsec_depth(args.infile, args.outfile, args.label)
