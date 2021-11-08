import os
import numpy as np


def get_stations(specfemdir):

    # The name of the filtered stations file
    station_file = os.path.join(specfemdir, "DATA", "STATIONS_FILTERED")

    # Read file line for line
    stations = np.genfromtxt(
        station_file, delimiter=None, dtype=None, encoding=None,
        names=('station', 'network', 'latitude', 'longitude', 'elevation',
               'burial'))

    return stations
