

def read_parfile(filename: str) -> dict:
    """Reads in a ``Par_file`` and returns a dictionary containing the
    parameters.

    Args:
        filename (str): Path to Par_file

    Returns:
        dict: dictionary with all parameters

    Last modified: Lucas Sawade, 2020.09.18 19.00 (lsawade@princeton.edu)
    """

    pars: dict = {}

    with open(filename, 'r') as f:
        for line in f.readlines():
            if line[0] == "#":
                pass
            elif line in ['\n', '\r\n']:
                pass
            elif '=' in line:
                keysec, valsec = line.split('=')[:2]
                key = keysec.split()[0]
                val = valsec.split('#')[0].split()[0]

                if val == '.true':
                    pars[key] = True

                elif val == '.false.':
                    pars[key] = False

                elif val.isnumeric():
                    pars[key] = int(val)

                else:
                    try:
                        pars[key] = float(val.replace(
                            'D', 'E').replace('d', 'e'))

                    except Exception:
                        pars[key] = val
            else:
                pass
    return pars
