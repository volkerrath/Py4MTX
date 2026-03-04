#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce a site list (name, coordinates, elevation) from EDI files.

Output formats are tailored for WALDIM, FEMTIC, or general use.

@author: sb & vr dec 2019
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

import os
import sys
import csv
import inspect

import numpy as np

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
PY4MTX_DATA = os.environ["PY4MTX_DATA"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

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
COORDS = "utm"
EPSG = None

DELIM = ","
WHAT_FOR = "wal"     # Options: "wal", "femtic", "kml"

if "wal" in WHAT_FOR:
    DELIM = " "
    COORDS = "latlon"

WORK_DIR = "/home/vrath/MT_Data/waldim/"
EDI_DIR = WORK_DIR + "/edi_jc/"

print(" Edifiles read from: %s" % EDI_DIR)

if "wal" in WHAT_FOR.lower():
    CSV_FILE = EDI_DIR + "Sitelist_waldim.txt"
elif "fem" in WHAT_FOR.lower():
    CSV_FILE = EDI_DIR + "Sitelist_femtic.txt"
else:
    CSV_FILE = EDI_DIR + "Sitelist.txt"
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
    sitelist = csv.writer(f, delimiter=DELIM)

    if "wal" in WHAT_FOR.lower():
        sitelist.writerow(["Sitename", "Latitude", "Longitude"])
        sitelist.writerow([ns, " ", " "])

    for sitenum, filename in enumerate(edi_files):
        print("reading data from: " + filename)
        name, _ = os.path.splitext(filename)
        file_i = EDI_DIR + filename
        edi_dict = load_edi(file_i, drop_invalid_periods=True)

        lat = edi_dict["lat"]
        lon = edi_dict["lon"]
        elev = edi_dict["elev"]

        if "utm" in COORDS.lower():
            if EPSG is not None:
                easting, northing = utl.proj_latlon_to_utm(
                    latitude=lat, longitude=lon, utm_zone=EPSG
                )
            else:
                sys.exit("make sitelist: utm required, but no EPSG given! Exit.")

        if "wal" in WHAT_FOR:
            sitelist.writerow([name, lat, lon])
        elif "fem" in WHAT_FOR:
            sitelist.writerow([name, lat, lon, elev, sitenum])
        else:
            sitelist.writerow([name, lat, lon, elev])
