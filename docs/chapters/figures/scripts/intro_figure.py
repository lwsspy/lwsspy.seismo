"""
This script creates figure 1 (a,b,c) of the gcmt3d article.
"""
import os
import lwsspy as lpy
import matplotlib.pyplot as plt
from cartopy.crs import Mollweide

# Get the GCMT3D data directory
datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Create base figure
fig = plt.figure(figsize=(13, 3.0))
plt.subplots_adjust(left=0.01, right=0.99,
                    bottom=0.3, top=0.99, wspace=0.05)


# Events
# Get catalog
gcmt_cat = lpy.seismo.CMTCatalog.load(os.path.join(datadir, "gcmtcatalog.pkl"))
filtered_cat = gcmt_cat.filter(
    mindict=dict(moment_magnitude=5.7),
    maxdict=dict(moment_magnitude=7.5))

# Plot Events
evax = plt.subplot(131, projection=Mollweide(central_longitude=0.0))
lmaps.plot_map(zorder=-1)
filtered_cat.plot(ax=evax)
lplt.plot_label(evax, "a)", location=1, box=False, dist=0.0)

# Stations
# Get stations
xml_name = "gcmt3d_station.xml"
invfile = os.path.join(datadir, xml_name)
inv = lpy.seismo.read_inventory(invfile)

# Plot Stations
stax = plt.subplot(132, projection=Mollweide(central_longitude=0.0))å
lmaps.plot_map(zorder=-1)
lpy.seismo.plot_inventory(inv, ax=stax, markersize=5, cmap='Set1')
# Set legend fontsizes
legendfontsize = "x-small"
title_fontsize = "small"
plt.legend(loc='upper center', title='Networks', frameon=False,
           bbox_to_anchor=(0.5, 0), numpoints=1, scatterpoints=1,
           fontsize=legendfontsize, title_fontsize=title_fontsize,
           handletextpad=0.2,  # norderaxespad=-2.5, borderpad=0.5,
           labelspacing=0.2, handlelength=1.0, ncol=7,
           columnspacing=1.0, bbox_transform=stax.transAxes)
lplt.plot_label(stax, "b)", location=1, box=False, dist=0.0)

# Velocity model
vname = "GLAD_M25_z100km_vpv.dat"
vfile = os.path.join(datadir, vname)

vax = plt.subplot(133, projection=Mollweide(central_longitude=0.0))
cax = vax.inset_axes(bounds=[0.1, -0.16, 0.8, 0.04])
_, cbar = lplt.plot_specfem_xsec_depth(
    vfile, ax=vax, cax=cax, depth=100.0)
cax.set_title("$v_{P_V}$ [km/s]", fontsize='small')
plt.xticks(fontsize="x-small")

lplt.plot_label(vax, "c)", location=1, box=False, dist=0.0)

plt.savefig(os.path.join(lbase.DOCFIGURES, "gcmt3d",
                         "events_stations_model.pdf"))
plt.show(block=True)
