from typing import Optional
import multiprocessing as mp
from functools import partial
from obspy import Stream, Inventory
from .window import window_on_stream
from ...utils.timer import Timer
from ...utils.multiqueue import multiwrapper
from ..process.split_stream_inv import split_stream_inv


def queue_multiwindow_stream(
        obsd: Stream,
        synt: Stream,
        windowdict,
        nproc: int = 4,
        verbose: bool = False):
    """Uses the multiprocessing module to multiprocess your python stream.
    Extremely simple:
    1. Split stream into chunks for each processor, where each chunk contains 
       full stations so it's possible to rotate.
    2. Sends the processes to the processing queue and let's them do their
       thing.       

    Parameters
    ----------
    obsd : Stream
        Observed data Stream
    synt : Stream
        Synthetic data Stream
    windowdict : dict
        processing dictionary. See ``lwsspy.process_stream`` for what can go in 
        there. It should contain ``inventory``
    nprocs : int, optional
        number of processors, by default 4
    verbose : bool
        flag whether to print some statements

    Returns
    -------
    Stream
        windowed observed stream


    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.06.13 17.50

    """
    if verbose:
        print("Split stream")
    # Split the stream into different chunks
    split_obsd, split_synt, _ = split_stream_inv(
        obsd, windowdict['station'], synt=synt, nprocs=nproc)

    # Create process wise args
    args = [(_obsd, _synt) for _obsd, _synt in zip(split_obsd, split_synt)]

    if verbose:
        print(args)

    with Timer():
        if verbose:
            print("Enter multiwrapper")

        # multiprocess stream
        results = multiwrapper(
            window_on_stream,
            args=args,
            kwargs=[windowdict for _ in range(len(args))],
            verbose=verbose)

        if verbose:
            print("Get results")

        # Create empty stream
        windowed_stream = Stream()

        # Populate stream with results
        for _res in results:
            windowed_stream += _res

    return windowed_stream
