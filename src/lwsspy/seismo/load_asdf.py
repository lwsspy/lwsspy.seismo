from pyasdf import ASDFDataSet
from obspy import Stream, Inventory


def load_asdf(filename: str, no_event=False):
    """Takes in a filename of an asdf file and outputs event, inventory,
    and stream with the traces. Note that this is only good for asdffiles
    with one set of traces event and stations since the function will get the
    first/only waveform tag from the dataset

    Parameters
    ----------
    filename : str
        ASDF filename. "somethingsomething.h5"
    no_event : bool, optional
        set to true if you only want the stream and the inventory
        returned, by default False

    Returns
    -------
    (event, stream, inventory) or (stream, inventory)
        Output data in obspy format

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.10.05 16.30

    """

    ds = ASDFDataSet(filename)

    # Create empty streams and inventories
    inv = Inventory()
    st = Stream()

    # Get waveform tag
    tag = list(ds.waveform_tags)[0]
    for station in ds.waveforms.list():
        try:
            st += getattr(ds.waveforms[station], tag)
            inv += ds.waveforms[station].StationXML
        except Exception as e:
            print(e)

    # Choose not to load an event from the asdf file (pycmt3d's event doesn't
    # output an event...)
    if not no_event:
        ev = ds.events[0]
        del ds

        return ev, inv, st
    else:
        del ds
        return inv, st
