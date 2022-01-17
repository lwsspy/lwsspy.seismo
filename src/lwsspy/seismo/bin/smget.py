#!/usr/bin/env python
'''
smget CMTCODE, e.g. smget C202106062316A
smget CMTCODE NET STA, e.g. smget C201606261117A II ARU

Downloads simulations from the ShakeMovie website

Originally taken from fjsimons-at-alum.mit.edu, 01/13/2021, and translated to
python.

'''

from subprocess import check_call
import socket

from sys import argv, exit
import os


def bin():

    # Where is the website?
    IP = '128.112.172.121'  # ???
    NS = 'global.shakemovie.princeton.edu'
    PR = 'https://'

    # The CMT code
    # Check if argument is given
    if len(argv) == 1:
        print(
            "\nYou need to provide a CMTCODE. Usage: smget CMTCODE"
            "e.g. smget C202106062316A\n\n",
            __doc__)
        exit()
    else:
        CMT = argv[1]

    # Get network if needed
    network = argv[2] if len(argv) == 4 else None

    # The station we want
    station = argv[3] if len(argv) == 4 else None

    # Flags identifying the "channels"
    sim = ['1d', '3d']
    # What we know their extensions are
    ext = ['modes', 'sem']
    # What we know their components are
    cmp = ['LX', 'MX']
    # dirs we want them in
    odir = ['MODES', 'SEM']

    # Where you will put it? If SMDATA exists put there else makedir
    mydata = os.getenv(
        "SMDATA",
        os.path.join(os.environ["HOME"], "SEISDATA", "SHAKEMOVIE_DATA")
    )
    # Current Host (--> use curl on Mac)
    current_host = socket.gethostname()

    # Get the files!
    for _sim, _cmp, _ext, _dir in zip(sim, cmp, ext, odir):
        # Destination filename
        fname = f'{CMT}_{_sim}.sac.tar.gz'
        # Makes the URL and performs the query

        # The below line used to work with http but ShakeMovie got an update to https and it broke
        # curl -o $fname ''$PR''$NS'/shakemovie/dl?evid='$CMT'&product='$SIM[$index]''

        # So now we change it as we did in MCMS2EVT
        evtquery = f"{PR}{NS}/shakemovie/dl?evid={CMT}&product={_sim}"

        # Curl or wget depending on host
        if current_host in ['geo-lsawade19']:
            actquery = f"curl -L -o {fname} '{evtquery}'"
            print(actquery)
        else:
            actquery = f"wget -q {evtquery} -O {fname}"

        # Actually downloading
        check_call(actquery, shell=True)

        # Check if file was downloaded.
        if not os.path.exists(fname):

            print(f"{fname} was not downloaded. Skipping extraction")
            continue

        else:
            # If the file was downloaded check whether there was an issue.
            try:
                with open(fname, 'r') as f:
                    dl_error = 'The ShakeMovie Server Has Encountered an Error' \
                        in f.read()
                if dl_error:
                    print(f'Download error. {CMT} does not seem to exist, or there'
                          f'was a different error.\n$ cat {fname}\n'
                          f'for webpage response details.')
                    continue
            except:
                pass

        # If download passed create the necessary directories
        # If you didn't have it, will make it
        myddir = os.path.join(mydata, odir)
        mydcmt = os.path.join(myddir, CMT)

        if not os.path.exists(mydcmt):
            os.makedirs(mydcmt)

        # Unpack
        if (network is not None) and (station is not None):

            # Make the destination
            fpart = f'{network}.{station}.{_cmp}'
            lpart = f'{_ext}.sac'

            # Turns out the components are E, N, and Z
            wewant = " ".join([f'{fpart}E.{lpart}',
                               f'{fpart}N.{lpart}',
                              f'{fpart}Z.{lpart}'])

            # Extract the station etc of interest

            check_call(f'tar -xvzf {fname} {wewant}', shell=True)

            # Now actually put it there
            check_call(f'mv {wewant} {mydcmt}', shell=True)

        else:
            # Extract the all stations to the directory
            check_call(
                f'tar -xvzf {fname} -C {mydcmt}',
                shell=True)

        # Remove downloaded file
        # os.remove(fname)


if __name__ == "__main__":
    bin()
