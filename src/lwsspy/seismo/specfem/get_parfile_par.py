from .read_parfile import read_parfile


def get_parfile_par(parfile: str, parameter: str):
    """Reads parfile intor dictionary, then return the parameter in question.
    This, of course, requires some understanding of the parameters contained
    in the Parfile.

    Parameters
    ----------
    parfile : str
        parfile location
    parameters : str
        parameter to be returned

    Returns
    -------
    bool or str or int or float 
        the return depends on the parameter

    Notes
    -----

    :Author:
        Lucas Sawade (sawade@princeton.edu)

    :Last Modified:
        2021.11.05 10.30


    """

    # Read Parfile
    pf = read_parfile(parfile)

    # Get Param and return
    return pf[parameter]
