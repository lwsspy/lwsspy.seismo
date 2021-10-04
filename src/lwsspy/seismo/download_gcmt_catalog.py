"""
Python script retrieving the full Global CMT catalog from 1976 to 2020
  - first  download full catalog from 1976 to 2017
  - second download monthly catalogs from 2018 to 2020
The reason for doing this instead of using obspy fdsn clien and IRIS
is because we not only want to retrieve the location but also focal
mechanisms.
----------------------------------------------------------------------
Result is a big ndk file named full_catalog.ndk
----------------------------------------------------------------------
Adapted to Python from S. Beller's Bash implementation
Written by S. Beller, 2020-12-19
"""

import os
import datetime as dt
from urllib.request import urlopen
import lwsspy as lpy


def download_gcmt_catalog():
    # Get catalog from 1976 to 2017
    url_cat = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec17.ndk"
    catalog_filename = os.path.join(lpy.base.DOWNLOAD_CACHE, "gcmt.ndk")

    # Download the catalog
    lpy.utils.print_action(f"Downloading {url_cat}")
    lpy.shell.downloadfile(url_cat, catalog_filename)

    # Get monthly catalog
    ext = '.ndk'
    link = 'https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/'

    thisyear = dt.datetime.now().year
    thismonth = dt.datetime.now().month

    with open(catalog_filename, "a") as catalogfile:

        for year in range(2018, dt.datetime.now().year + 1):

            yy = f"{year}"[-2:]

            for month in ["jan", "feb", "mar", "apr", "may", "jun",
                          "jul", "aug", "sep", "oct", "nov", "dec"]:

                if (year == thisyear) \
                        and (month == thismonth):
                    break
                else:

                    url_monthly = f"{link}{year}/{month}{yy}{ext}"
                    lpy.utils.print_action(f"Downloading {url_monthly}")

                    try:
                        catalogfile.write(
                            urlopen(url_monthly).read().decode('utf-8'))
                    except Exception as e:
                        print(e)
