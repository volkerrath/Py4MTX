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
import xarray as xr
from mtpy.core.mt import MT
import mtpy.core.mt as mt

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import util as utl
import mtproc as mtp
from version import versionstrg


version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

# Define the path to your EDI-files:
PY4MTX_DATA =  "/home/vrath/MT_Data/"
WorkDir = PY4MTX_DATA+"/Ubaye_best/"
EdiDir_in = WorkDir+"/edis/"
print(" Edifiles read from: %s" % EdiDir_in)
EdiDir_out =  WorkDir+"/edis_clean/"
print(" Edifiles written to: %s" % EdiDir_out)
if not os.path.isdir(EdiDir_out):
    print(" File: %s does not exist, but will be created" % EdiDir_out)
    os.mkdir(EdiDir_out)

String_out = ""
SearchStrng = ".edi"

bad = 0.

# No changes required after this line!

# Construct list of EDI-files:

# Construct list of EDI-files:

edi_files = mtp.get_edi_list(EdiDir_in, fullpath=False)
ns = np.size(edi_files)

# Enter loop:

for filename in edi_files:
    print("\n Reading data from " + EdiDir_in + filename)
    name, ext = os.path.splitext(filename)

# Create an MT object

    file_i = EdiDir_in + filename
    mt_obj= MT()
    mt_obj.read(file_i)
    # print(mt_obj.station_metadata)

    lat = mt_obj.station_metadata.location.latitude
    lon = mt_obj.station_metadata.location.longitude
    elev = mt_obj.station_metadata.location.elevation
    print(" site %s at :  % 10.6f % 10.6f % 8.1f" % (name, lat, lon, elev ))

    new_mt_obj = mtp.clean_data(mt_obj=mt_obj, bad=bad)

    file_o = EdiDir_out+filename.replace(".edi","_clean.edi")
    new_mt_obj.write(file_o,
                     latlon_format="dd",
                     longitude_format="LONG")
    print("Data written to ", file_o, "\n")
