from typing import Optional
import multiprocessing as mp
from functools import partial
from obspy import Stream, Inventory
from .process import process_stream
from ...utils.timer import Timer
from .split_stream_inv import split_stream_inv


def multiprocess_stream(
        st: Stream,
        processdict,
        nprocs=4,
        pool: Optional[mp.Pool] = None):
    """Uses the multiprocessing module to multiprocess your python stream.
    Extremely simple:
    1. Split stream into chunks for each processor, where each chunk contains 
       full stations so it's possible to rotate.
    2. Uses a pool, its map, and partial to map the different sets of functions
       to the processors.

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
    pool: multiprocessing.Pool
        can take in an existing processing pool, default None

    Returns
    -------
    Stream
        processed stream


    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.04.17 17.50

    """

    # Split the stream into different chunks
    splitstream, _ = split_stream_inv(st, processdict['inventory'])

    with Timer():

        if pool is not None:
            results = pool.map(
                partial(process_stream, **processdict), splitstream)
        else:
            with mp.Pool(processes=nprocs) as pool:
                results = pool.map(
                    partial(process_stream, **processdict), splitstream)

        # Create empty stream
        processed_stream = Stream()

        # Populate stream with results
        for _res in results:
            processed_stream += _res

    return processed_stream
