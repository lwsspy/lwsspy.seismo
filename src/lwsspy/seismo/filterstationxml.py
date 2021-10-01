from obspy import read_inventory
from copy import deepcopy


def filterstationxml(xml_in: str, xml_out: str, **kwargs):
    """Filters a station xml given the kwargs parsed to the
    `Inventory.select()` method.

    Parameters
    ----------
    xml_in : filename
        StationXML file to be filter by filter criteria
    xml_out : [type]
        output StationXML file to be written from filtered inventory

    Last modified: Lucas Sawade, 2020.09.18 19.00 (lsawade@princeton.edu)
    """

    # Read stationxml
    inv = read_inventory(xml_in)

    # filter Inventtory
    inv_f = inv.select(**kwargs)

    # write new stationxml
    inv_f.write(xml_out, format="STATIONXML")
