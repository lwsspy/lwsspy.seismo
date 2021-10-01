from os import path as p
from typing import Union
from ...shell.cp import cp
from .createsimdir import createsimdir


def cmt2simdir(cmtfilename: str, specfemdir: str, outputdir: str = "./",
               specfem_dict: Union[dict, None] = None,
               stationsfile: Union[str, None] = None):
    """Takes in ``CMTSOLUTION`` file and specfem directory and creates specfem
    simulation directory.
    Uses the ``cmtfilename`` to create new simulation directory in the
    output directory.

    Args:
        cmtfilename (str):
            Path to CMTSOLUTION
        specfemdir (str):
            Path to specfem dir
        outputdir (str, optional):
            Output Directory. Defaults to "./".
        specfem_dict (Union[dict, None], optional):
            Optional dictionary. Defaults to None which gives the following
            copy dirtree dict from the ``createsimdir`` function

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
        stationsfile (Union[str, None], optional):
            Define ``STATIONS`` file to be written to the simulation directory.
            Default is ``None``, which won't overwrite the specfem's 
            ``STATIONS`` file.

    Returns:
        None

    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)

    """

    # Get CMT basename
    cmtname = p.basename(cmtfilename)

    # Create output dirtree root
    root = p.join(outputdir, cmtname)

    # Copy directory to new destination
    createsimdir(specfemdir, root, specfem_dict=specfem_dict)

    # Overwrite cmt solution
    cp(cmtfilename, p.join(root, "DATA", "CMTSOLUTION"), ow=True)

    # Overwrite STATIONS file if given
    if stationsfile is not None:
        cp(stationsfile, p.join(root, "DATA", "STATIONS"), ow=True)
