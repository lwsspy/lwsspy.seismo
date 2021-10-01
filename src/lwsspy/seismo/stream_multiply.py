from obspy import Stream


def stream_multiply(st: Stream, factor: float):
    """Acts on stream and multiplies included data by a `factor`

    Parameters
    ----------
    st : Stream
        stream to be processed
    factor : float
        factor to multiply traces with

    Last modified: Lucas Sawade, 2020.10.30 14.00 (lsawade@princeton.edu)
    """

    # Loop over traces
    for tr in st:
        tr.data *= factor
