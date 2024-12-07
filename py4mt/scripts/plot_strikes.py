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
surveyname = "enfield"

# Define the path to your EDI-files:
EdiDir = WorkDir
file_list = mtp.get_edi_list(EdiDir)
ns = len(file_list)

# Define the path to your MTCollection file:
Collection = WorkDir+"/enfield_collection.h5"

FromEdis = True
if FromEdis:
    print(" Edifiles read from: %s" % EdiDir)

    dataset = mtp.make_data(edirname=EdiDir,
                        collection=Collection,
                        metaid="enfield",
                        survey=surveyname,
                        savedata=True,
                        utm_epsg=EPSG
                        )

    # dataset = mtp.make_collection(
    #                     collection=Collection,
    #                     metaid="enfield",
    #                     survey=surveyname,
    #                     returndata=True,
    #                     utm_epsg=EPSG
    #                     )

else:
    with MTCollection() as mtc:
        mtc.open_collection(Collection)
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
PDFCatalog = True
PDFCatalogName  = "Enfield_strikes.pdf"
if not ".pdf" in PlotFmt:
    PDFCatalog= False
    print("No PDF catalog because no pdf output!")


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
    strike_plot_all.save_plot(PltDir+"StrikesAllData"+fmt, fig_dpi=600)


strike_plot_dec = dataset.plot_strike(plot_type=1,
                                      print_stats=True,
                                      text_pad=.005,
                                      plot_pt = True,
                                      plot_tipper = True,
                                      plot_invariant = True,
                                      plot_orientation="v")
for fmt in PlotFmt:
    strike_plot_dec.save_plot(PltDir+"StrikesPerDec"+fmt, fig_dpi=600)

if PDFCatalog:
    pdf_list = []
    # catalog = PdfPages(PDFCatalogName)

# # # Loop over stations
    for sit in file_list:
        site, _ = os.path.splitext(os.path.basename(sit))
        data = dataset.get_subset([surveyname+"."+site])
        strike_plot_site = data.plot_strike()
        for fmt in PlotFmt:
            strike_plot_site.save_plot(PltDir+"Strikes_"+site+fmt, fig_dpi=600)
