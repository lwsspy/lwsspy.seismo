"""

This will not be executed as part of the pytest suite, but can be invoked by

.. code:: bash

    mpirun -np 4 python test_process_mpi.py

"""

import os
import psutil
from pprint import pprint
from obspy import read
import lwsspy.utils as lutils
from lwsspy.seismo import CMTSource
from lwsspy.seismo import read_inventory
from lwsspy.seismo.process.queue_multiprocess_stream import queue_multiprocess_stream
from lwsspy.seismo.process.multiprocess_stream import multiprocess_stream
from lwsspy.utils.io import read_yaml_file
from lwsspy.utils import reset_cpu_affinity

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
WAVEFORMS = os.path.join(DATADIR, "data", "observed.mseed")
STATIONS = os.path.join(DATADIR, "data", "stations.xml")
PROCESSFILE = os.path.join(SCRIPTDIR, "data", "process.yml")


def test_queue_multiprocessing():

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
        remove_response_flag=True,
        event_latitude=cmtsource.latitude,
        event_longitude=cmtsource.longitude,
        geodata=True)
    )

    # Load waveforms
    stream = read(WAVEFORMS)

    # Print stuff
    pprint(processdict)
    pprint(stream)

    # Reset
    # reset_cpu_affinity()

    with lutils.Timer():
        processed_stream_q = queue_multiprocess_stream(
            stream, processdict, nproc=4, verbose=True)
        print("queue")
        print(processed_stream_q.select(network="G", station="TAOE"))

    # with lpy.Timer():
    #     processed_stream_m = multiprocess_stream(stream, processdict, nprocs=20)
    #     print("map")
    #     print(processed_stream_m.select(network="G", station="TAOE"))

    # with lpy.Timer():
    #     processed_stream = lpy.process_stream(stream, **processdict)
    #     print("single cpu")
    #     print(processed_stream.select(network="G", station="TAOE"))


if __name__ == "__main__":
    print_affinity()
    test_queue_multiprocessing()
    print_affinity()
