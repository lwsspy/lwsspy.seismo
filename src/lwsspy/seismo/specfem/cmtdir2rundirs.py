from glob import glob
from os import path as p
from typing import Union

from .cmt2rundir import cmt2rundir
from ...shell.cp import cp


def cmtdir2rundirs(cmtdir: str, specfemdir: str,
                   specfem_dict: Union[dict, None] = None):
    """Wrapper arount ``cmt2simdir`` to process an entire directory

    Args:
        cmtdir (str):
            Directory of CMT solutions
        specfemdir (str):
            Path to specfem dir
        outputdir (str, optional):
            Output Directory. Defaults to "./".
        specfem_dict (Union[dict, None], optional):
            Optional dictionary. Defaults to None which gives the following
            copy dirtree dict from the ``createsimdir`` function:

            .. code:: python

                {
                    "bin": "link",
                    "DATA": {
                        "CMTSOLUTION": "file",
                        "Par_file": "file",
                        "STATIONS": "file"
                    },
                    "DATABASES_MPI": "link",
                    "OUTPUT_FILES": "dir"
                }

    Returns:
        None

    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)

    """

    # Get list of cmt files
    cmtfiles = glob(p.join(cmtdir, "*"))

    print("Number of CMT files: %d" % len(cmtfiles))

    for i, _file in enumerate(cmtfiles):
        print(f"#{i+1:0>5}/{len(cmtfiles)}:{_file:_>50}")
        rundir = p.join(specfemdir, f"run{i+1:0>4}")
        cmt2rundir(_file, specfemdir, rundir,
                   specfem_dict=specfem_dict)

        srctxt = p.join(rundir, 'source.txt')
        with open(srctxt, 'w') as f:
            f.write(p.basename(_file))

        cp(_file, p.join(rundir, p.basename(_file)))
