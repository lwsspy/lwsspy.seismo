import os
import instaseis
import tarfile
from ... import shell as lsh
from ... import base as lbase


def get_prem20s():

    # Define locations
    instadir = os.path.join(lbase.DOWNLOAD_CACHE,
                            "instaseis")
    filetar = os.path.join(instadir, "PREM20s.tar.gz")
    dbdir = os.path.join(instadir, "PREM20s")

    # Download url
    url = "http://www.geophysik.uni-muenchen.de/~krischer/instaseis/20s_PREM_ANI_FORCES.tar.gz"

    # Create directory if it doesn't exist
    if os.path.exists(instadir) is False:
        os.makedirs(instadir)

    # Check if downloaded already
    if os.path.exists(dbdir) is False:
        lsh.downloadfile(url, filetar)
        with tarfile.open(filetar) as tar:
            tar.extractall(path=dbdir)

    return instaseis.open_db(dbdir)
