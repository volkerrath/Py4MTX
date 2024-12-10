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
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
Rotate imedance tensor (Z) and tipper (T)
by potentially different angles into NS/EW coordinate system

@author: sb & vr Jan 2020

"""

# Import required modules

import os
import sys
import numpy as np

from mtpy.core.mt import MT
import mtpy.core.mt as mt

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import util
from version import versionstrg


version, _ = versionstrg()
titstrng = util.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

# Define the path to your EDI-files:
EdiDir_in = PY4MTX_ROOT+"/work/orig/"


PY4MTX_DATA =  "/home/vrath/MT_Data/"
WorkDir = PY4MTX_DATA+"/Ubaye_best/"
EdiDirIn = WorkDir+"/edis/"

print(" Edifiles read from: %s" % EdiDir_in)
EdiDir_out =  EdiDirIn+"/with_pt/"
print(" Edifiles written to: %s" % EdiDir_out)
if not os.path.isdir(EdiDir_out):
    print(" File: %s does not exist, but will be created" % EdiDir_out)
    os.mkdir(EdiDir_out)

String_out = ""
SearchStrng = ".edi"

Declination = 0. # 2.68   #E
DecDeg = True


# No changes required after this line!

# Construct list of EDI-files:
edi_files = []
# input EDI-file. As the actual call is dine using the "eval" function,
files = os.listdir(EdiDir_in)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        if SearchStrng in entry:
            edi_files.append(entry)
ns = np.size(edi_files)

# Enter loop:

for filename in edi_files:
    print("\n Reading data from " + EdiDir_in + filename)
    name, ext = os.path.splitext(filename)

# Create an MT object

    file_i = EdiDir_in + filename
    tf_obj = MT()
    tf_obj.read(file_i)

    lat = tf_obj.station_metadata.location.latitude
    lon = tf_obj.station_metadata.location.longitude
    elev = tf_obj.station_metadata.location.elevation
    print(" site %s at :  % 10.6f % 10.6f % 8.1f" % (name, lat, lon, elev ))
    # tf_obj.write(file_i.replace(".edi","_check.edi"))

    new_tf_obj = tf_obj.rotate(Declination, inplace=False)


# Write a new edi file:
    # tf_obj.write("newfile.edi", latlon_format='dd', longitude_format='LONG')

    file_out =EdiDir_out+name+String_out+ext
    print(" Writing data to "+file_out)
    if DecDeg:
        tf_obj.write(file_out, latlon_format='dd')
    else:
        tf_obj.write(file_out)
