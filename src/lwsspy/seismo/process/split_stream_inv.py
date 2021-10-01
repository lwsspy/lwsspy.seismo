from typing import Optional
from obspy import Stream, Inventory


def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]


def split_stream_inv(
        obsd: Stream,
        inv: Inventory,
        synt: Optional[Stream] = None,
        nprocs=4):
    """Depending on the number of cores provided it splits the stream into
    station sets, so that rotation is easily done.

    Parameters
    ----------
    st : Stream
        Input Stream
    inv : Inventory
        Input inventory
    nprocs : int, optional
        number of preocessors to be used, by default 4
    """

    # Split up stations into station chunks
    stations = [x.split()[0] for x in inv.get_contents()['stations']]
    sstations = split(stations, nprocs)

    # Split up traces into chunks containing full stations
    sobsd, sinv = [], []
    if synt is not None:
        ssynt = []

    for _stalist in sstations:
        # Create substreams and inventories
        subinv = Inventory()
        obsd_substream = Stream()

        if synt is not None:
            synt_substream = Stream()

        for _sta in _stalist:
            network, station = _sta.split(".")
            subinv += inv.select(network=network, station=station)
            obsd_substream += obsd.select(network=network, station=station)
            if synt is not None:
                synt_substream += synt.select(network=network, station=station)

        sinv.append(subinv)
        sobsd.append(obsd_substream)
        if synt is not None:
            ssynt.append(synt_substream)

    if synt is not None:
        return sobsd, ssynt, sinv
    else:
        return sobsd, sinv
