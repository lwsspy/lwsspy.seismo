import os
from .. import utils as lutil
from .. import base as lbase


def load_1976_2004_mag():
    return lutil.loadxy_csv(os.path.join(lbase.GCMT_DATA, '1976_2004.csv'))


def load_2004_2010_mag():
    return lutil.loadxy_csv(os.path.join(lbase.GCMT_DATA, '2004_2010.csv'))


def load_cum_mag():
    return lutil.loadxy_csv(os.path.join(lbase.GCMT_DATA, 'cum_mag.csv'))


def load_num_events():
    return lutil.loadxy_csv(os.path.join(lbase.GCMT_DATA, 'no_events.csv'))
