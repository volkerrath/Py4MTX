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


PY4MT_ROOT = os.environ["PY4MT_ROOT"]
PY4MT_DATA = os.environ["PY4MT_DATA"]

mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import util
from version import versionstrg

PY4MT_DATA = os.environ["PY4MT_DATA"]

version, _ = versionstrg()
titstrng = util.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


dialect = "unix"
delim = " "
whatfor = "waldim"
if  "wal" in whatfor:
    delim = " "


# Define the path to your EDI-files and for the list produced
EdiDir = PY4MT_ROOT+"/work/orig/"
EdiDir = "/home/vrath/MT_Data/Enfield/"
print(" Edifiles read from: %s" % EdiDir)
CSVFile = EdiDir + "Sitelist.dat"
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

# Outputfile (e. g., for WALDIM analysis)

with open(CSVFile, "w") as f:

    sitelist = csv.writer(f, delimiter=delim)
    if "wal" in whatfor:
        sitelist.writerow(["Sitename", "Latitude", "Longitude"])
        sitelist.writerow([ns, " ", " "])

# Loop over edifiles:

    for filename in edi_files:
        print("reading data from: " + filename)
        name, ext = os.path.splitext(filename)
        file_i = EdiDir + filename

# Create MT object
        mt_obj = MT()
        mt_obj.read(file_i)
        lat = mt_obj.station_metadata.location.latitude
        lon = mt_obj.station_metadata.location.longitude
        elev = mt_obj.station_metadata.location.elevation

        # sitename = mt_obj.station
        if "wal" in whatfor:
            sitelist.writerow([name, lat, lon])
        else:
            sitelist.writerow([name, lat, lon, elev])
