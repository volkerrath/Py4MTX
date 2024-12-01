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

# from mtpy.analysis.geometry import dimensionality
#from mtpy.imaging.plotstrike import PlotStrike

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import contextily as cx

from mtpy import MT , MTData, MTCollection


PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/modules/", PY4MTX_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
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

PY4MTX_DATA =  "/home/vrath/MT_Data/"
edi_dir = "Enfield/"
print(" Edifiles read from: %s" % edi_dir)
PlotFmt = [".png", ".pdf", ".svg"]

EPSG = 32629

ReadEdi = True
# No changes re32629quired after this line!

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


# Outputfile (e. g., for WALDIM analysis)

# """
# Determine graphical parameter.
# => print(plt.style.available)
# """

plt.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["savefig.facecolor"] = "none"

# # Loop over edifiles:
# n3d = 0
# n2d = 0
# n1d = 0
# nel = 0
sit = 0


# mtd = MTData()
# mtd.open_collection(edi_dir+"enfield_data.h5")
mtc = MTCollection()
mtc.open_collection(edi_dir+"enfield_collection.h5")

for filename in edi_files:
    sit = sit + 1
    print("reading data from: " + filename)
    name, ext = os.path.splitext(filename)
    file_i = filename

# Create MT object

    mt_obj = MT()
    mt_obj.read(file_i)
    mt_obj.survey_metadata.id = "enfield"
    mtc.add_tf(mt_obj)


#     lon = mt_obj.lon
#     lat = mt_obj.lat
#     elev = mt_obj.elev
#     east = mt_obj.east
#     north = mt_obj.north
#     freq = mt_obj.Z.freq
# # use the phase tensor to determine which frequencies are 1D/2D/3D
#     dim = dimensionality(z_object=mt_obj.Z,
#                          skew_threshold=5,
#                          # threshold in skew angle (degrees) to determine if
#                          # data are 3d
#                          # threshold in phase ellipse eccentricity to determine
#                          # if data are 2d (vs 1d)
#                          eccentricity_threshold=0.1
#                          )
#     dims = np.vstack([freq,dim])
#     print(np.shape(dims))
#     np.savetxt(fname =name+"_dim.dat",fmt="%12.5f  %3i",X =dims.T)

#     print("dimensionality:")
#     nel = nel + np.size(dim)
#     n1d = n1d + sum(map(lambda x: x == 1, dim))
#     n2d = n2d + sum(map(lambda x: x == 2, dim))
#     n3d = n3d + sum(map(lambda x: x == 3, dim))

#     strikeplot = PlotStrike(fn_list=[file_i],
#                             plot_type=1,
#                             plot_tipper='yr')
#     #
#     strikeplot.save_plot(name+"_Strikes.png",
#                           file_format='.png',
#                           fig_dpi=600)
#     print("  number of undetermined elements = " +
#           str(nel - n1d - n2d - n3d) + "\n")
#     print("  number of 1-D elements = " + str(sum(map(lambda x: x == 1, dim))) +
#           "  (" + str(round(100 * sum(map(lambda x: x == 1, dim)) / nel)) + "%)")
#     print("  number of 2-D elements = " + str(sum(map(lambda x: x == 2, dim))) +
#           "  (" + str(round(100 * sum(map(lambda x: x == 2, dim)) / nel)) + "%)")
#     print("  number of 3-D elements = " + str(sum(map(lambda x: x == 3, dim))) +
#           "  (" + str(round(100 * sum(map(lambda x: x == 3, dim)) / nel)) + "%)")

mtc.working_dataframe = mtc.master_dataframe.loc[mtc.master_dataframe.survey == "enfield"]
mtc.utm_crs = 32629
mtd = mtc.to_mt_data()
mtc.close_collection()


stations_plot = mtd.plot_stations(pad=.005)
for fmt in PlotFmt:
    stations_plot.save_plot(edi_dir+"AllSites"+fmt, fig_dpi=600)

strike_plot_all = mtd.plot_strike()
for fmt in PlotFmt:
    stations_plot.save_plot(edi_dir+"StrikesAllData"+fmt, fig_dpi=600)


strike_plot_dec = mtd.plot_strike(plot_type=1, print_stats=True)
for fmt in PlotFmt:
    stations_plot.save_plot(edi_dir+"StrikesPerDec"+fmt, fig_dpi=600)

pt_map = mtd.plot_phase_tensor_map(
    plot_tipper="yri",
    cx_source=cx.providers.Esri.NatGeoWorldMap,
    ellipse_size=.02,
    arrow_size=.05
)


# print("\n\n\n")
# print("number of sites = " + str(sit))
# print("total number of elements = " + str(nel))
# print("  number of undetermined elements = " +
#       str(nel - n1d - n2d - n3d) + "\n")
# print("  number of 1-D elements = " + str(n1d) +
#       "  (" + str(round(100 * n1d / nel)) + "%)")
# print("  number of 2-D elements = " + str(n2d) +
#       "  (" + str(round(100 * n2d / nel)) + "%)")
# print("  number of 3-D elements = " + str(n3d) +
#       "  (" + str(round(100 * n3d / nel)) + "%)")
