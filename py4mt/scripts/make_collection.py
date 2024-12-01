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


mtp.make_collection(edirname=edi_dir,
                    cfilename=colfile,
                    metaid="Enfield",
                    survey="enfield",
                    utm_epsg=32629
                    )

print("MT Collection written to:", edi_dir+"enfield_collection.h5")

if MakeMap:
    with MTCollection() as mtc:
        mtc.open_collection(colfile)
        # mtc.working_dataframe = mtc.working_dataframe.query('station.str.startswith("en")')
        mtd = mtc.to_mt_data()

    stations_plot = mtd.plot_stations(pad=.005)
    for fmt in PltFmt:
        stations_plot.save_plot(MapFile+fmt, fig_dpi=600)
