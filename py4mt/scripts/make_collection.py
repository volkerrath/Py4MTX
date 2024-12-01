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
from sys import exit as error

import numpy as np
from mtpy import MT , MTData, MTCollection


PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/modules/", PY4MTX_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import mtproc as mtp
import modem as mod
import util as utl

from version import versionstrg

rng = np.random.default_rng()
nan = np.nan  # float("NaN")
blank = 1.e-30 # np.nan
rhoair = 1.e17

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


cm = 1/2.54  # centimeters in inches

# Define the path to your EDI-files and for the list produced
edi_dir = "/home/vrath/MT_Data/Enfield/"
print(" Edifiles read from: %s" % edi_dir)
colfile = edi_dir+"Enfield_collection.h5"

MakeMap = True
if MakeMap:
    MapFile = edi_dir+"Enfield_map"
    PltFmt =  [".png", ".pdf", ".svg"]



# Construct list of edi-files:

edi_files = []
files = os.listdir(edi_dir)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(edi_dir+entry)
ns = np.size(edi_files)
if ns ==0:
    error("No edi files found in "+edi_dir+"! Exit.")


mtd =mtp.make_collection(edirname=edi_dir,
                    cfilename=colfile,
                    metaid="enfield_data",
                    dataout=True,
                    survey="enfield_survey",
                    utm_epsg=32629
                    )
# # # Loop over edifiles:
# # n3d = 0
# # n2d = 0
# # n1d = 0
# # nel = 0
# sit = 0

# mtc = MTCollection()
# mtc.open_collection(edi_dir+"enfield_collection.h5")

# for filename in edi_files:
#     sit = sit + 1
#     print("reading data from: " + filename)
#     name, ext = os.path.splitext(filename)
#     file_i = filename

# # Create MT object

#     mt_obj = MT()
#     mt_obj.read(file_i)
#     mt_obj.survey_metadata.id = "enfield"
#     mtc.add_tf(mt_obj)

# mtc.working_dataframe = mtc.master_dataframe.loc[mtc.master_dataframe.survey == "enfield"]
# mtc.utm_crs = 32629
# mtd = mtc.to_mt_data()
# mtc.close_collection()

print("MT Collection written to:", edi_dir+"enfield_collection.h5")

if MakeMap:
    stations_plot = mtd.plot_stations(pad=.005)
    for fmt in PltFmt:
        stations_plot.save_plot(MapFile+fmt, fig_dpi=600)
