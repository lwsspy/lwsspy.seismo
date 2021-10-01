# Internal
from .process import process_stream
from ..source import CMTSource

# External
from obspy import Stream, Inventory
from typing import Union
from copy import deepcopy


def process_wrapper(st: Stream, event: CMTSource, paramdict: dict,
                    inv: Union[Inventory, None] = None,
                    observed: bool = True):
    """Fixes start and endtime in the dictionary

    Parameters
    ----------
    stream : Stream
        stream to be processed
    event : CMTSource
        event
    paramdict : dict
        parameterdictionary
    inv : Inventory, optional
        Station data, if you want to remove the station response, default None
    observed: bool, optional
        if you want to remove the station response and only use a single 
        process parameter file, default True

    Returns
    -------
    dict
        processparameter dict

    

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.01.06 11.00
    """

    newdict = deepcopy(paramdict)
    rstart = newdict.pop("relative_starttime")
    rend = newdict.pop("relative_endtime")
    newdict.update({
        "starttime": event.cmt_time + rstart,
        "endtime": event.cmt_time + rend,
    })
    newdict.update({"remove_response_flag": observed})

    return process_stream(st, event_latitude=event.latitude,
                          event_longitude=event.longitude,
                          inventory=inv, **newdict)
