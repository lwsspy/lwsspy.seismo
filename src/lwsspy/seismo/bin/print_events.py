"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.06 16.00

"""

from os import path, listdir
from sys import argv
from ..source import CMTSource


def bin():
    """

    Usage:

        seismo-print-events <path/to/eventdir>

    This script calls a python function that just prints all events in a 
    directory to the terminal. Just for info.

    """
    # Get args or print usage statement
    if (len(argv) != 2) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        eventdir = argv[1]

    for file in listdir(eventdir):

        filepath = path.join(eventdir, file)

        cmt = CMTSource.from_CMTSOLUTION_file(filepath)
        print()
        print(f'{cmt.eventname:>30}')
        print('-' * 30)
        print()
        print(cmt)
