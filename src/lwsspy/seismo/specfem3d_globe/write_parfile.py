

def write_parfile(pars: dict, filename: str):
    """Writes parameter dictionary to file

    Parameters
    ----------
    pars : dict
        Dictionary with Specfem Parameters
    filename : str
        Filename to write the Parameters to.

    Last modified: Lucas Sawade, 2020.10.18 19.00 (lsawade@princeton.edu)

    """

    # Get Parfile content
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Use modified Parameter dictionary to set parameters.
    for i, line in enumerate(lines):
        if '=' in line:
            keysec = line.split('=')[0]
            key = keysec.split()[0].strip()

            if key in pars and pars[key] is not None:
                # convert float or bool to str
                val = pars[key]

                if isinstance(val, bool):
                    val = f'.{str(val).lower()}.'

                elif isinstance(val, float):
                    if len('%f' % val) < len(f'{val}'):
                        val = '%fd0' % val

                    else:
                        val = f'{val}d0'

                lines[i] = f'{keysec}= {val}'

    with open(filename, 'w') as f:
        for line in lines:
            if "#" in line:
                pass
            elif line == "":
                pass
            else:
                f.write(line)
                f.write('\n')
