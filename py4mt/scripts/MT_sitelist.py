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
mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


dialect = "unix"
delim = " "
whatfor = "waldim"
if  "wal" in whatfor:
    delim = " "

PY4MT_DATA = os.environ["PY4MT_DATA"]
# Define the path to your EDI-files and for the list produced
edi_dir = PY4MT_DATA+"/Opf/2023/edi/"
# r"/home/vrath/Desktop/MauTopo/MauEdi/"
# r"/media/vrath/MT/Ireland/Northwest_CarboniferousBasin/MT_DATA/EDI/"
print(" Edifiles read from: %s" % edi_dir)
csv_file = edi_dir + "Sitelist.dat"
print("Writing data to file: " + csv_file)


# No changes required after this line!

# Construct list of edi-files:

edi_files = []
files = os.listdir(edi_dir)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)
ns = np.size(edi_files)

# Outputfile (e. g., for WALDIM analysis)

with open(csv_file, "w") as f:

    sitelist = csv.writer(f, delimiter=delim)
    if "wal" in whatfor:
        sitelist.writerow(["Sitename", "Latitude", "Longitude"])
        sitelist.writerow([ns, " ", " "])

# Loop over edifiles:

    for filename in edi_files:
        print("reading data from: " + filename)
        name, ext = os.path.splitext(filename)
        file_i = edi_dir + filename

# Create MT object

        mt_obj = MT(file_i)
        lon = mt_obj.lon
        lat = mt_obj.lat
        elev = mt_obj.elev
        east = mt_obj.east
        north = mt_obj.north
        # sitename = mt_obj.station
        if "wal" in whatfor:
            sitelist.writerow([name, lat, lon])
        else:
            sitelist.writerow([name, lat, lon, elev])
