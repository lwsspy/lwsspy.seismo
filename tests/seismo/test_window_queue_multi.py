"""

This will not be executed as part of the pytest suite, but can be invoked by

.. code:: bash

    mpirun -np 4 python test_process_mpi.py

"""

import os
from obspy import read, read_events
import obspy
from lwsspy.seismo import read_inventory
from lwsspy.seismo.window.queue_multiwindow_stream import queue_multiwindow_stream
from lwsspy.utils.io import read_yaml_file

try:
    import mpi4py
    mpimode = True
except ImportError as e:
    print(e)
    mpimode = False

SCRIPT = os.path.abspath(__file__)
SCRIPTDIR = os.path.dirname(__file__)
DATADIR = os.path.join(SCRIPTDIR, "data", "testdatabase", "C200811240902A")
CMTFILE = os.path.join(DATADIR, "C200811240902A_gcmt")
OBSERVED = os.path.join(DATADIR, "data", "processed_observed.mseed")
SYNTHETIC = os.path.join(DATADIR, "data", "processed_synthetic.mseed")
STATIONS = os.path.join(DATADIR, "data", "stations.xml")
WINDOWFILE = os.path.join(SCRIPTDIR, "data", "window.yml")


def test_queue_multiwindow():

    # Loading and fixing the processin dictionary
    windowdict = read_yaml_file(WINDOWFILE)
    # Wrap window dictionary
    wrapwindowdict = dict(
        station=read_inventory(STATIONS),
        event=read_events(CMTFILE)[0],
        config_dict=windowdict,
        _verbose=False
    )

    # Load waveforms
    observed = read(OBSERVED)
    synthetic = read(SYNTHETIC)

    # Window the thing
    windowed_stream = queue_multiwindow_stream(
        observed, synthetic, wrapwindowdict, nproc=4)

    # Check windowing
    for tr in windowed_stream:
        print(len(tr.stats.windows))


if __name__ == "__main__":

    test_queue_multiwindow()
