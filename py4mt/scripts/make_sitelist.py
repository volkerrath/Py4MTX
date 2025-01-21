#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: "1.5"
#       jupytext_version: 1.11.3
# ---

"""

This script produces a site list containing site names,
coordinates and elevations, e. g., for WALDIM analysis.

@author: sb & vr dec 2019
"""

# Import required modules

import os
import sys
import csv
from mtpy.core.mt import MT
import numpy as np

from mtpy import MT , MTData, MTCollection


PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
PY4MTX_DATA = os.environ["PY4MTX_DATA"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import util as utl
from version import versionstrg

PY4MTX_DATA = os.environ["PY4MTX_DATA"]

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


dialect = "unix"
delim = ","
whatfor = "femtic"
# whatfor = "kml"
if  "wal" in whatfor:
    delim = " "


# Define the path to your EDI-files and for the list produced
PY4MTX_DATA = "/home/vrath/Py4MTX/work/results_ploting/"
WorkDir = PY4MTX_DATA
EdiDir =  WorkDir + "/edi_files/"

print(" Edifiles read from: %s" % EdiDir)

if  "wal" in whatfor:
    CSVFile = WorkDir + "Sitelist_waldim.txt"
elif "fem" in whatfor:
    CSVFile = WorkDir + "Sitelist_femtic.txt"
else:
    CSVFile = WorkDir + "Sitelist.txt"
print("Writing data to file: " + CSVFile)

# No changes required after this line!

# Construct list of edi-files:

edi_files = []
files = os.listdir(EdiDir)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)
ns = np.size(edi_files)
edi_files = sorted(edi_files)

# Outputfile (e. g., for WALDIM analysis)

with open(CSVFile, "w") as f:

    sitelist = csv.writer(f, delimiter=delim)
    if "wal" in whatfor:
        sitelist.writerow(["Sitename", "Latitude", "Longitude"])
        sitelist.writerow([ns, " ", " "])

# Loop over edifiles:
    sitenum=0

    for filename in edi_files:
        print("reading data from: " + filename)
        name, ext = os.path.splitext(filename)
        file_i = EdiDir + filename
        sitenum = sitenum + 1

# Create MT object
        mt_obj = MT()
        mt_obj.read(file_i)
        lat = mt_obj.station_metadata.location.latitude
        lon = mt_obj.station_metadata.location.longitude
        elev = mt_obj.station_metadata.location.elevation



        # sitename = mt_obj.station
        if "wal" in whatfor:
            sitelist.writerow([name, lat, lon])
        elif "fem" in whatfor:
            sitelist.writerow([name, lat, lon, elev, sitenum])
        else:
            sitelist.writerow([name, lat, lon, elev])
