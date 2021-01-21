"""
This script generates a figure with a station map using functions
from lwsspy. Internet required as the station map is downloaded.

Map created for GCMT3D paper.

:Author:
    Lucas Sawade (lsawade@princeton.edu)

:Last modifided:
    Lucas Sawade 2021.01.14 13.00
"""

import os
from obspy import UTCDateTime
import lwsspy as lpy
import matplotlib.pyplot as plt
lpy.updaterc()

# Networks used in the Global CMT Project
networks = ["II", "IU", "IC", "G", "MN", "GE", "CU"]
networksstring = ",".join(networks)

# number of legend columns
ncol = len(networks)


# Define file
xml_name = "gcmt3d_station.xml"
invfile = os.path.join(lpy.DOCFIGURESCRIPTDATA, xml_name)

# Write inventory after downloading
if os.path.exists(invfile):
    inv = lpy.read_inventory(invfile)
else:
    # Download metadata
    start = UTCDateTime(1976, 1, 1)
    end = UTCDateTime.now()
    duration = end - start

    # Download the data
    inv = lpy.download_data(start, duration=duration, network=networksstring,
                            station=None, dtype='stations')

    # Write it to the script data directory
    inv.write(invfile, "STATIONXML")

# Create Figure
plt.figure(figsize=(6, 3.5))
plt.subplots_adjust(bottom=0.05, top=1.0, left=0.01, right=0.99,
                    wspace=0.0)

# Create map axes
ax = lpy.map_axes(proj='moll')

# Plot continents
lpy.plot_map()

# Plot inventory
lpy.plot_inventory(inv, markersize=7, cmap='Set1')

# Create Legend
plt.legend(loc='lower center', frameon=False, fancybox=False,
           numpoints=1, scatterpoints=1, fontsize='small',
           borderaxespad=-2.5, borderpad=0.5, handletextpad=0.2,
           labelspacing=0.2, handlelength=1.0, ncol=ncol,
           columnspacing=1.0)

plt.savefig(os.path.join(lpy.DOCFIGURES, "station_map.svg"), dpi=300,
            transparent=True)
plt.savefig(os.path.join(lpy.DOCFIGURES, "station_map.pdf"), transparent=True)
