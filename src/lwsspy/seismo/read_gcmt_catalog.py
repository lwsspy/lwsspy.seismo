import os
from typing import Union, List
from obspy import Catalog, read_events
import lwsspy as lpy
import time


def read_gcmt_catalog(force: bool = False):

    # filename
    gcmt_filename = os.path.join(lpy.DOWNLOAD_CACHE, "gcmt.ndk")

    if force or not os.path.exists(gcmt_filename):
        # Download catalog
        t0 = time.time()
        lpy.download_gcmt_catalog()
        t1 = time.time()
        print(f"Downloading took {t1 - t0} seconds.")

    # Read the catalog
    lpy.print_action("Reading the catalog")
    t0 = time.time()
    cat = read_events(gcmt_filename)
    t1 = time.time()
    print(f"Reading took {t1 - t0} seconds.")

    return cat
