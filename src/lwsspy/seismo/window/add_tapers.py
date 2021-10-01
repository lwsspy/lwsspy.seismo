import obspy
from scipy import signal


def add_tapers(observed: obspy.Stream, taper_type: str = "tukey",
               alpha: float = 0.25, verbose: bool = False):
    """Attaches tapers to all Trace.Stats in Stream using preexisting 
    Stats.windows.

    Parameters
    ----------
    observed : obspy.Stream
        observed data
    taper_type : str, optional
        taper type, until now only tukey is supported, by default "tukey"
    alpha : float, optional
        fraction of traces that the tapering should occupy totally, that is,
        at both ends together, by default 0.25

    Last modified: Lucas Sawade, 2020.09.28 19.00 (lsawade@princeton.edu)
    """

    if taper_type == "tukey":
        taper = signal.tukey

    for tr in observed:
        # Create empty list of tapers
        tr.stats.tapers = []

        try:
            for win in tr.stats.windows:
                length = win.right - win.left
                tr.stats.tapers.append(taper(length, alpha=alpha))
        except Exception as e:
            if verbose:
                print(
                    f"Not able to read windows at"
                    f"{tr.stats.network}.{tr.stats.station}."
                    f"{tr.stats.location}.{tr.stats.channel}: {e}")
