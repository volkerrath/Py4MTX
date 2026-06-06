#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce a site list (name, coordinates, elevation) from EDI files.

Output formats are tailored for WALDIM, FEMTIC, or general use.

@author: sb & vr dec 2019
Cleanup: 4 Mar 2026 by Claude (Anthropic)
Modified: 2026-05-23 — use utm_zone_from_latlon / latlon_to_utm_zn after
    reading each EDI; FEMTIC output now includes easting and northing after
    sitenum; Claude Sonnet 4.6 (Anthropic)
"""

import os
import sys

from pathlib import Path
import csv
import inspect

import numpy as np

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
PY4MTX_DATA = os.environ["PY4MTX_DATA"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import util as utl
from version import versionstrg
from data_proc import load_edi

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
DELIM = ","
WHAT_FOR = "fem"     # Options: "wal", "femtic", "kml"


#: UTM zone override — None = auto-derived from each site's lat/lon.
#: Set to an integer (e.g. 19) to force a fixed zone for all sites.
UTM_ZONE_OVERRIDE = None

#WORK_DIR = "/home/vrath/Py4MTX/py4mt/data/rto/ubinas//"
WORK_DIR = "/home/vrath/Py4MTX/py4mt/data/rto/misti//"
EDI_DIR = WORK_DIR + "/edi/"

print(" Edifiles read from: %s" % EDI_DIR)

CSV_FILE = EDI_DIR + "site.dat"
print("Writing data to file: " + CSV_FILE)

# =============================================================================
#  Build file list and write site list
# =============================================================================
edi_files = sorted([
    entry for entry in os.listdir(EDI_DIR)
    if entry.endswith(".edi") and not entry.startswith(".")
])
ns = len(edi_files)

with open(CSV_FILE, "w") as f:

    if "wal" in WHAT_FOR.lower():
        sitelist = csv.writer(f, delimiter=" ")
        sitelist.writerow(["Sitename", "Latitude", "Longitude"])
        sitelist.writerow([ns, " ", " "])
    else:
        sitelist = csv.writer(f, delimiter=DELIM)
        # sitelist.writerow(
            # ["Sitename", "Latitude", "Longitude", "Site#", "Easting", "Northing"])
        # sitelist.writerow([ns, " ", " "])

    for sitenum, filename in enumerate(edi_files):
        print("reading data from: " + filename)

        name, _ = os.path.splitext(filename)
        file_i = EDI_DIR + filename
        edi_dict = load_edi(file_i, drop_invalid_periods=True)

        lat = edi_dict["lat"]
        lon = edi_dict["lon"]
        elev = edi_dict["elev"]

        zone, northern = utl.utm_zone_from_latlon(
            lat, lon, override=UTM_ZONE_OVERRIDE)
        easting, northing = utl.latlon_to_utm_zn(lat, lon, zone, northern)
        easting = np.around(easting,1)
        northing = np.around(northing,1)
        print(f"  zone {zone}{'N' if northern else 'S'}  "
              f"E={easting:.1f}  N={northing:.1f}")

        if "wal" in WHAT_FOR.lower():
            sitelist.writerow([name, lat, lon])
        elif "fem" in WHAT_FOR.lower():
            sitelist.writerow(
                [name, lat, lon, elev, sitenum, easting, northing])
        elif "kml" in WHAT_FOR.lower():
            sitelist.writerow(
                [name, lat, lon, elev, sitenum, easting, northing])
        else:
            sitelist.writerow(
                [name, lat, lon, elev, sitenum, easting, northing])
