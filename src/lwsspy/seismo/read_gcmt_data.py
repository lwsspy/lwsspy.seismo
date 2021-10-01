import os
import lwsspy as lpy


def load_1976_2004_mag():
    return lpy.loadxy_csv(os.path.join(lpy.GCMT_DATA, '1976_2004.csv'))


def load_2004_2010_mag():
    return lpy.loadxy_csv(os.path.join(lpy.GCMT_DATA, '2004_2010.csv'))


def load_cum_mag():
    return lpy.loadxy_csv(os.path.join(lpy.GCMT_DATA, 'cum_mag.csv'))


def load_num_events():
    return lpy.loadxy_csv(os.path.join(lpy.GCMT_DATA, 'no_events.csv'))
