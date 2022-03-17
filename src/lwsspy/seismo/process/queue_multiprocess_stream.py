from typing import Optional
import multiprocessing as mp
from functools import partial
from obspy import Stream, Inventory
from .process import process_stream
from lwsspy.utils.timer import Timer
from lwsspy.utils.multiqueue import multiwrapper
from .split_stream_inv import split_stream_inv


def queue_multiprocess_stream(
        st: Stream,
        processdict,
        nproc: int = 4,
        verbose: bool = False) -> Stream:
    """Uses the multiprocessing module to multiprocess your python stream.
    Extremely simple:
    1. Split stream into chunks for each processor, where each chunk contains 
       full stations so it's possible to rotate.
    2. Sends the processes to the processing queue and let's them do their
       thing.

    **Warning**

    As of now I need the inventory, the split function should be modified to 
    get the stations from the stream and not the inventory.

    Parameters
    ----------
    st : Stream
        Input Stream
    processdict : dict
        processing dictionary. See ``lwsspy.process_stream`` for what can go in 
        there. It should contain ``inventory``
    nprocs : int, optional
        number of processors, by default 4
    verbose : bool
        flag whether to print some statements

    Returns
    -------
    Stream
        processed stream


    Notes
    -----

    :Author:f
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.06.13 17.50

    """
    if verbose:
        print("Split stream")
    # Split the stream into different chunks
    splitstream, _ = split_stream_inv(st, processdict['inventory'],
                                      nprocs=nproc)
    # Make argument tuples
    splitstream = [(st, ) for st in splitstream if len(st) > 0]

    if verbose:
        print(splitstream)

    with Timer():
        if verbose:
            print("Enter multiwrapper")

        # multiprocess stream
        results = multiwrapper(
            process_stream,
            args=splitstream,
            kwargs=[processdict for _ in range(len(splitstream))],
            verbose=verbose)

        if verbose:
            print("Get results")

        # Create empty stream
        processed_stream = Stream()

        # Populate stream with results
        for _res in results:
            processed_stream += _res

    return processed_stream
