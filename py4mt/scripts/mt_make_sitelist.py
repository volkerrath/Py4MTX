#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce a site list (name, coordinates, elevation) from EDI files.

Output formats are tailored for WALDIM, FEMTIC, or general use.

@author: sb & vr dec 2019
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
Coords = "utm"
EPSG = None

delim = ","
whatfor = "wal"     # Options: "wal", "femtic", "kml"

if "wal" in whatfor:
    delim = " "
    Coords = "latlon"

WorkDir = "/home/vrath/MT_Data/waldim/"
EdiDir = WorkDir + "/edi_jc/"

print(" Edifiles read from: %s" % EdiDir)

if "wal" in whatfor.lower():
    CSVFile = EdiDir + "Sitelist_waldim.txt"
elif "fem" in whatfor.lower():
    CSVFile = EdiDir + "Sitelist_femtic.txt"
else:
    CSVFile = EdiDir + "Sitelist.txt"
print("Writing data to file: " + CSVFile)

# =============================================================================
#  Build file list and write site list
# =============================================================================
edi_files = sorted([
    entry for entry in os.listdir(EdiDir)
    if entry.endswith(".edi") and not entry.startswith(".")
])
ns = len(edi_files)

with open(CSVFile, "w") as f:
    sitelist = csv.writer(f, delimiter=delim)

    if "wal" in whatfor.lower():
        sitelist.writerow(["Sitename", "Latitude", "Longitude"])
        sitelist.writerow([ns, " ", " "])

    for sitenum, filename in enumerate(edi_files):
        print("reading data from: " + filename)
        name, _ = os.path.splitext(filename)
        file_i = EdiDir + filename
        edi_dict = load_edi(file_i, drop_invalid_periods=True)

        lat = edi_dict["lat"]
        lon = edi_dict["lon"]
        elev = edi_dict["elev"]

        if "utm" in Coords.lower():
            if EPSG is not None:
                easting, northing = utl.proj_latlon_to_utm(
                    latitude=lat, longitude=lon, utm_zone=EPSG
                )
            else:
                sys.exit("make sitelist: utm required, but no EPSG given! Exit.")

        if "wal" in whatfor:
            sitelist.writerow([name, lat, lon])
        elif "fem" in whatfor:
            sitelist.writerow([name, lat, lon, elev, sitenum])
        else:
            sitelist.writerow([name, lat, lon, elev])
