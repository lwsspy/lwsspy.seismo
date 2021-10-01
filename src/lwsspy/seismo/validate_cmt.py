from .source import CMTSource


def validate_cmt(cmt: CMTSource):
    """Checks whether source has valid location parameters

    Args:
        cmt (CMTSource): cmt

    Raises:
        ValueError: if depth is smaller than 0
        ValueError: if latitude is outside [-90, 90]

    Last modified: Lucas Sawade, 2020.09.22 11.00 (lsawade@princeton.edu)
    """

    if cmt.depth_in_m <= 0.0:
        raise ValueError("Depth(m): %f < 0" % (cmt.depth_in_m))

    if cmt.latitude < -90.0 or cmt.latitude > 90.0:
        raise ValueError("Error of cmt latitude: {}".format(cmt.latitude))

    if cmt.longitude < -180.0:
        cmt.longitude += 360.
    elif cmt.longitude > 180.0:
        cmt.longitude -= 360
