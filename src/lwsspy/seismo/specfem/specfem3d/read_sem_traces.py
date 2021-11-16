import os
import numpy as np
from glob import glob

from numpy.core.fromnumeric import trace


def read_sem_traces(specfemdir, unit='d'):

    # Glob all traces
    files = glob(os.path.join(specfemdir, "OUTPUT_FILES", f"*.sem{unit}"))
    # Station.Network
    NetStaChaTTr = []


    for _file in files:
        f = os.path.basename(_file)
        network = f.split('.')[0]
        station = f.split('.')[1]
        channel = f.split('.')[2]
        
        t, tr = np.loadtxt(_file).T

        NetStaChaTTr.append((network, station, channel, t, tr))

    # Get shapes
    netdtype = type(NetStaChaTTr[0][0])
    stadtype = type(NetStaChaTTr[0][1])
    chadtype = type(NetStaChaTTr[0][2])
    tshape = NetStaChaTTr[0][3].shape
    tdtype = NetStaChaTTr[0][3].dtype
    trshape = NetStaChaTTr[0][4].shape
    trdtype = NetStaChaTTr[0][4].dtype

    dtypes = np.dtype([
        ('network', 'U10'),
        ('station', 'U10'),
        ('channel', 'U10'),
        ('t', tdtype, tshape),
        ('tr', trdtype, trshape)
    ])

    return np.array(NetStaChaTTr, dtype=dtypes)
    
