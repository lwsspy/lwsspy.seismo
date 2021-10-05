import os
from copy import deepcopy
import lwsspy as lpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.ticker import StrMethodFormatter
import cartopy
from cartopy.crs import PlateCarree, Mollweide
import numpy as np

lplt.updaterc()

# GCMT Catalog main data to plot
npzfilename = os.path.join(lbase.DOCFIGURESCRIPTDATA, "gcmt.npz")

if os.path.exists(npzfilename):
    # Load data
    data = np.load(npzfilename)
    qlat = data['qlat']
    qlon = data['qlon']
    qdepth = data['qdepth']
    qmoment = data['qmoment']
    depth = data['depth']
    moment = data['moment']
    latitude = data['latitude']
    longitude = data['longitude']

else:
    # Read Catalog
    cat = lpy.seismo.read_gcmt_catalog()

    # Create list of things that is easy to handle (not obspy event.)
    cmts = []
    for event in cat:
        cmts.append(lpy.seismo.CMTSource.from_event(event))

    latitude = []
    longitude = []
    depth = []
    moment = []
    for cmt in cmts:
        latitude.append(cmt.latitude)
        longitude.append(cmt.longitude)
        depth.append(cmt.depth_in_m/1000.0)
        moment.append(cmt.moment_magnitude)
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    depth = np.array(depth)
    moment = np.array(moment)

    # Interpolate moment/depth heat maps
    snn = lpy.SphericalNN(latitude, longitude)
    llat = np.linspace(-90, 90, 361)
    llon = np.linspace(-180, 180, 721)
    qlat, qlon = np.meshgrid(llat, llon)
    gcmt_interpolator = snn.interpolator(
        qlat, qlon, maximum_distance=1.25, k=200)
    qdepth = gcmt_interpolator(depth)
    qmoment = gcmt_interpolator(moment)

    # Save data to npz
    np.savez(
        npzfilename, **dict(latitude=latitude, longitude=longitude,
                            depth=depth, moment=moment,
                            qlat=qlat, qlon=qlon, qdepth=qdepth,
                            qmoment=qmoment)
    )
    del snn
    del cat

###########################   SCATTER #########################################

# Create png
_, ax, _, _ = lpy.seismo.plot_quakes(latitude, longitude, depth, moment,
                                     cmap='rainbow_r', yoffsetlegend=0.05)
outnamepng = os.path.join(
    lbase.DOCFIGURES, "gcmt3d", "gcmt_depth_moment.png")
plt.savefig(outnamepng)
plt.close()

# Create vectors
_, ax, _, _ = lpy.seismo.plot_quakes(latitude, longitude, depth, moment,
                                     cmap='rainbow_r', yoffsetlegend=0.05)
outnamepdf = os.path.join(
    lbase.DOCFIGURES, "gcmt3d", "gcmt_depth_moment.pdf")
plt.savefig(outnamepdf)

# Scatter submap
_, ax, _, _ = lpy.seismo.plot_quakes(latitude, longitude, depth, moment,
                                     cmap='rainbow_r', yoffsetlegend=0.0)
fig = ax.figure
fig.set_size_inches(6, 10)
plt.subplots_adjust(left=0.025, right=0.975, bottom=0.03, top=1.0)
ax.set_extent([160, 185, -50, -10])

outnamepng = os.path.join(lbase.DOCFIGURES, "gcmt3d",
                          "gcmt_depth_moment_tonga.png")
plt.savefig(outnamepng, dpi=300)

outnamepdf = os.path.join(lbase.DOCFIGURES, "gcmt3d",
                          "gcmt_depth_moment_tonga.pdf")
plt.savefig(outnamepdf)
plt.close()


##############################################################################
lsel = 5.75
hsel = 7.25
selection = (lsel <= moment) & (moment <= hsel)

rmoment = deepcopy(moment)
rmoment[hsel <= rmoment] = hsel
rmoment[rmoment <= lsel] = lsel


plt.figure(figsize=(5, 6.5))
plt.subplots_adjust(left=0.025, right=0.975, bottom=0.15, top=0.95)
ax1 = plt.subplot(211, projection=Mollweide(central_longitude=-150.0))
ax1.set_global()
lmaps.plot_map(zorder=-1)
lpy.seismo.plot_quakes(latitude, longitude, depth, rmoment, ax=ax1,
                       cmap='rainbow_r', legend=False)
lplt.plot_label(ax1, "a)", box=False)
lplt.plot_label(ax1, f"N: {len(depth)}", location=2, box=False,
                fontdict=dict(fontsize='x-small'))
plt.title("GCMT Events")
ax2 = plt.subplot(212, projection=Mollweide(central_longitude=-150.0))
ax2.set_global()
lmaps.plot_map(zorder=-1)
lpy.seismo.plot_quakes(latitude[selection], longitude[selection],
                       depth[selection], moment[selection], ax=ax2,
                       cmap='rainbow_r')
lplt.plot_label(ax2, "b)", box=False)
lplt.plot_label(ax2, f"N: {len(depth[selection])}", location=2, box=False,
                fontdict=dict(fontsize='x-small'))
plt.title(f"GCMT Events {lsel} $\leq M_w \leq$ {hsel}")


# Save as PNG
outnamepng = os.path.join(lbase.DOCFIGURES, "gcmt3d",
                          "gcmt_depth_moment_compare.png")
plt.savefig(outnamepng, dpi=300)

# Save as PDF
outnamepdf = os.path.join(lbase.DOCFIGURES, "gcmt3d",
                          "gcmt_depth_moment_compare.pdf")
plt.savefig(outnamepdf)

# Sideways ##################################################################


plt.figure(figsize=(10, 3.5))
plt.subplots_adjust(left=0.025, right=0.975,
                    bottom=0.15, top=0.95, wspace=0.05)
ax1 = plt.subplot(121, projection=Mollweide(central_longitude=-150.0))
ax1.set_global()
lmaps.plot_map(zorder=-1)
lpy.seismo.plot_quakes(latitude, longitude, depth, rmoment, ax=ax1,
                       cmap='rainbow_r', legend=True, xoffsetlegend=0.525)
lplt.plot_label(ax1, "a)", box=False)
lplt.plot_label(ax1, f"N: {len(depth)}", location=2, box=False,
                fontdict=dict(fontsize='x-small'))
plt.title("GCMT Events")
ax2 = plt.subplot(122, projection=Mollweide(central_longitude=-150.0))
ax2.set_global()
lmaps.plot_map(zorder=-1)
scatter, ax, l1, l2 = lpy.seismo.plot_quakes(
    latitude[selection], longitude[selection],
    depth[selection], moment[selection], ax=ax2,
    cmap='rainbow_r', legend=False)

lplt.plot_label(ax2, "b)", box=False)
lplt.plot_label(ax2, f"N: {len(depth[selection])}", location=2, box=False,
                fontdict=dict(fontsize='x-small'))

plt.title(f"GCMT Events {lsel} $\leq M_w \leq$ {hsel}")


# Save as PNG
outnamepng = os.path.join(lbase.DOCFIGURES, "gcmt3d",
                          "gcmt_depth_moment_compare_side.png")
plt.savefig(outnamepng, dpi=300)

# Save as PDF
outnamepdf = os.path.join(lbase.DOCFIGURES, "gcmt3d",
                          "gcmt_depth_moment_compare_side.pdf")
plt.savefig(outnamepdf)
