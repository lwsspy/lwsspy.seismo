"""

This will not be executed as part of the pytest suite, but can be invoked by

.. code:: bash

    mpirun -np 4 python test_process_mpi.py

"""

import os
from obspy import read, read_events
from lwsspy.seismo import read_inventory
from lwsspy.seismo.window.mpiwindowclass import MPIWindowStream
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


def manual_test_mpiwindowing():

    if mpimode:

        # Initialize MPIprocess class
        WC = MPIWindowStream()

        if WC.rank == 0:
            print("MPI MODE")
            print(f"RANK: {WC.rank}")

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

            # Print stuff
            WC.get_streams_and_windowdict(observed, synthetic, wrapwindowdict)

        WC.window()

        # if WC.rank == 0:
        #     for _tr in WC.windowed_stream:
        #         print(_tr.id)
        #         try:
        #             print(len(_tr.stats.windows))
        #         except Exception:
        #             print(_tr.stats)

    else:
        print("NOT MPI MODE")
        return


if __name__ == "__main__":

    manual_test_mpiwindowing()
