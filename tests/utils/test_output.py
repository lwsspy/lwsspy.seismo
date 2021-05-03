import lwsspy as lpy
from logging import Logger
import logging


def test_loggers():

    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    # create file handler and set level to debug
    fh = logging.FileHandler('example.log', mode='w')
    fh.setLevel(logging.DEBUG)

    # create formatter
    formatter = lpy.CustomFormatter()

    # add formatter to ch
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(sh)
    logger.addHandler(fh)

    # 'application' code
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')
