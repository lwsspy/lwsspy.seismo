from asyncio import constants
from os import path
import yaml


PROCF = path.join(path.dirname(path.abspath(__file__)),
                  'process', 'process.yml')
with open(PROCF, "rb") as fh:
    PROCD = yaml.load(fh, Loader=yaml.FullLoader)
