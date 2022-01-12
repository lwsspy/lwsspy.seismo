#!/usr/bin/env python
"""
Short script originally written by Congyue Cui. 

The script has to be run in the specdem directory. Given no command
line arguments or 'm' the script runs the mesher. To run the solver provide
's'. 

The script will automatically detect the necessary number of tasks needed for 
submission from the ``Par_file``

"""

from subprocess import check_call
from sys import argv


def bin():

    nprocs = 1

    with open('DATA/Par_file', 'r') as f:
        for line in f.readlines():
            if '=' in line:
                key, val = line.split('=')[:2]

                if key.replace(' ', '') in ('NCHUNKS', 'NPROC_XI', 'NPROC_ETA'):
                    if '#' in val:
                        val = val.split('#')[0]

                    nprocs *= int(val)

    if len(argv) <= 1 or argv[1] == 'm':
        check_call(f'jsrun -n {nprocs} bin/xmeshfem3D', shell=True)

    if len(argv) <= 1 or argv[1] == 's':
        check_call(f'jsrun -n {nprocs} -g 1 bin/xspecfem3D', shell=True)
