from typing import Optional
import multiprocessing as mp
from functools import partial
from obspy import Stream, Inventory
from .window import window_on_stream_wrapper
from ...utils.timer import Timer
from ..process.split_stream_inv import split_stream_inv


def multiwindow_stream(
        obsd: Stream,
        synt: Stream,
        windowdict: dict,
        nprocs=4,
        pool: Optional[mp.Pool] = None):
    """Uses the multiprocessing module to multiwindow your python streams.
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
    obsd : Stream
        Input Stream
    synt : Stream
        Input Stream
    windowdict : dict
        processing dictionary. See ``lwsspy.process_stream`` for what can go in 
        there. It should contain ``inventory``
    nprocs : int, optional
        number of processors, by default 4
    pool : multiprocessing.Pool


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
    split_obsd, split_synt, _ = split_stream_inv(
        obsd, windowdict['station'], synt=synt)

    # Create process wise args
    args = [(_obsd, _synt) for _obsd, _synt in zip(split_obsd, split_synt)]

    # Actual windowing
    with Timer():
        if pool is not None:
            results = pool.map(
                partial(window_on_stream_wrapper, **windowdict), args)
        else:
            with mp.Pool(processes=nprocs) as pool:
                results = pool.map(
                    partial(window_on_stream_wrapper, **windowdict), args)

        # Create empty stream
        windowed_stream = Stream()

        # Populate stream with results
        for _res in results:
            print(_res)
            windowed_stream += _res

    return windowed_stream
