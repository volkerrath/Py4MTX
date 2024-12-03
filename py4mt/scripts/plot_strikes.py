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


import getpass
import datetime

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mtpy import MT, MTCollection, MTData

import contextily as cx




PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import mtproc as mtp
import util as utl

from version import versionstrg

rng = np.random.default_rng()
nan = np.nan  # float("NaN")
blank = 1.e-30 # np.nan
rhoair = 1.e17

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")



PY4MTX_DATA =  "/home/vrath/MT_Data/"


EPSG = 32629
WorkDir = PY4MTX_DATA+"/Enfield/"

# Define the path to your EDI-files:
EdiDir = WorkDir

# Define the path to your MTCollection file:
CollFile = WorkDir+"/enfield_collection.h5"

FromEdis = True
if FromEdis:
    print(" Edifiles read from: %s" % EdiDir)
    dataset = mtp.make_collection(edirname=EdiDir,
                        collection="Collfile",
                        metaid="enfield",
                        survey="enfield",
                        returndata=True,
                        utm_epsg=EPSG
                        )
else:
    with MTCollection() as mtc:
        mtc.open_collection(CollFile)
        mtc.working_dataframe = mtc.master_dataframe.loc[mtc.master_dataframe.survey == "enfield"]
        mtc.utm_crs = EPSG
        dataset = mtc.to_mt_data()


# Define the path to your EDI-files and for the list produced

# Define the  path for saving  plots:
PltDir = WorkDir +"/plots/"
print(" Plots written to: %s" % PltDir)
if not os.path.isdir(PltDir):
    print(" File: %s does not exist, but will be created" % PltDir)
    os.mkdir(PltDir)

# Graphical paramter. Determine the plot formats produced,
# and the required resolution:

PlotFmt = [".png", ".pdf", ".svg"]
DPI = 600
PdfCatalog = True
if not ".pdf" in PlotFmt:
    PdfCatalog= False
    print("No PDF catalog because no pdf output!")
PdfCatalogName  = "ANN_data.pdf"


# No changes reuired after this line!

# """
# Determine graphical parameter. not working with py4mt?
# => print(plt.style.available)
# """
# cm = 1/2.54  # centimeters in inches

# plt.style.use("seaborn-v0_8-paper")
# mpl.rcParams["figure.dpi"] = 600
# mpl.rcParams["axes.linewidth"] = 0.5
# mpl.rcParams["savefig.facecolor"] = "none"




stations_plot = dataset.plot_stations(pad=.005)
for fmt in PlotFmt:
    stations_plot.save_plot(PltDir+"AllSites"+fmt, fig_dpi=600)

strike_plot_all = dataset.plot_strike()
for fmt in PlotFmt:
    stations_plot.save_plot(PltDir+"StrikesAllData"+fmt, fig_dpi=600)


strike_plot_dec = dataset.plot_strike(plot_type=1, print_stats=True)
for fmt in PlotFmt:
    stations_plot.save_plot(PltDir+"StrikesPerDec"+fmt, fig_dpi=600)

# pt_map = dataset.plot_phase_tensor_map(
#     plot_tipper="yri",
#     cx_source=cx.providers.Esri.NatGeoWorldMap,
#     ellipse_size=.02,
#     arrow_size=.05
# )

# # Loop over stations
sit = 0
