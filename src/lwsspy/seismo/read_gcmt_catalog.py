import os
from obspy import read_events
import time
from .. import base as lbase
from .. import utils as lutil
from .download_gcmt_catalog import download_gcmt_catalog


def read_gcmt_catalog(force: bool = False):
    """Reads cached gcmt catalog or, if not downloaded yet initiates the download.

    Parameters
    ----------
    force : bool, optional
        Enforces redownload of the catalog, by default False

    Returns
    -------
    obspy.Catalog
        Catalog of events in the gcmt catalog

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.10.05 16.30


    """

    # filename
    gcmt_filename = os.path.join(lbase.DOWNLOAD_CACHE, "gcmt.ndk")

    if force or not os.path.exists(gcmt_filename):
        # Download catalog
        t0 = time.time()
        download_gcmt_catalog()
        t1 = time.time()
        print(f"Downloading took {t1 - t0} seconds.")

    # Read the catalog
    lutil.print_action("Reading the catalog")
    t0 = time.time()
    cat = read_events(gcmt_filename)
    t1 = time.time()
    print(f"Reading took {t1 - t0} seconds.")

    return cat
