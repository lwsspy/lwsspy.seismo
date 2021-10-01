from typing import Union
from ...shell.copy_dirtree import copy_dirtree

def createsimdir(specfemdir: str, outputdirname: str,
                 specfem_dict: Union[dict, None] = None):
    """Creae simulation directory from specfem directory.

    Args:
        specfemdir (str):
            specfem3d_globe directory
        outputdirname (str):
            new simulation directory with link to specfem directory.
        specfem_dict (Union[dict, None], optional):
            Optional dictionary. Defaults to None which gives the following
            copy dirtree dict:

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

    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)
    """

    if specfem_dict is None:
        specfem_dict = {
            "bin": "link",
            "DATA": {
                "CMTSOLUTION": "file",
                "Par_file": "file",
                "STATIONS": "file"
            },
            "DATABASES_MPI": "link",
            "OUTPUT_FILES": "dir"
            }

    # Copy Specfem directory
    copy_dirtree(specfemdir, outputdirname, dictionary=specfem_dict, ow=True)
