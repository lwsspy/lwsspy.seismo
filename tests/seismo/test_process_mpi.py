"""

This will not be executed as part of the pytest suite, but can be invoked by

.. code:: bash

    mpirun -np 4 python test_process_mpi.py

"""

import os
from pprint import pprint
from obspy import read
from lwsspy import CMTSource
from lwsspy import read_inventory
from lwsspy import process_stream
from lwsspy.seismo.process.mpiprocessclass import MPIProcessStream
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
WAVEFORMS = os.path.join(DATADIR, "data", "synthetic.mseed")
STATIONS = os.path.join(DATADIR, "data", "stations.xml")
PROCESSFILE = os.path.join(SCRIPTDIR, "data", "process.yml")


def manual_test_mpiprocessing():

    if mpimode:

        # Initialize MPIprocess class
        PC = MPIProcessStream()

        if PC.rank == 0:
            print("MPI MODE")
            print(f"RANK: {PC.rank}")

            # CMTSource
            cmtsource = CMTSource.from_CMTSOLUTION_file(CMTFILE)

            # Loading and fixing the processin dictionary
            processdict = read_yaml_file(PROCESSFILE)
            processdict["inventory"] = read_inventory(STATIONS)
            processdict["starttime"] = cmtsource.cmt_time \
                + processdict["relative_starttime"]
            processdict["endtime"] = cmtsource.cmt_time \
                + processdict["relative_endtime"]
            processdict.pop("relative_starttime")
            processdict.pop("relative_endtime")
            processdict.update(dict(
                remove_response_flag=False,
                event_latitude=cmtsource.latitude,
                event_longitude=cmtsource.longitude,
                geodata=False)
            )

            # Load waveforms
            stream = read(WAVEFORMS)

            # Print stuff
            pprint(processdict)
            pprint(stream)

            PC.get_stream_and_processdict(stream, processdict)

        PC.process()

        # if PC.rank == 0:
        #     PC.processed_stream.write(
        #         "processed_synthetic.mseed", format="MSEED")

    else:
        print("NOT MPI MODE")
        return


if __name__ == "__main__":

    manual_test_mpiprocessing()
