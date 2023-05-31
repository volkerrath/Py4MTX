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
from sys import exit as error
import csv

from mtpy.core.mt import MT
from mtpy.analysis.geometry import dimensionality
from mtpy.imaging.plotstrike import PlotStrike

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

cm = 1/2.54  # centimeters in inches

# Define the path to your EDI-files and for the list produced
edi_dir = r"/home/vrath/work/MT_Data/Opf/2023/edi/"
print(" Edifiles read from: %s" % edi_dir)

plotfile = r"/home/vrath/work/MT_Data/Opf/2023/dim/"
PlotFmt = [".png", ]

FilesOnly = True

DimFile = "Opf2023_dims.dat"
# No changes required after this line!

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

"""
Determine graphical parameter.
=> print(plt.style.available)
"""

plt.style.use("seaborn-paper")
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["savefig.facecolor"] = "none"
Fontsize = 8
Labelsize = Fontsize
Titlesize = 8
Fontsizes = [Fontsize, Labelsize, Titlesize]

Linewidths= [0.5]
Markersize = 4

"""
For just plotting to files, choose the cairo backend (eps, pdf, ,png, jpg...).
If you need to see the plots directly (plots window, or jupyter), simply
comment out the following line. In this case matplotlib may run into
memory problems ager a few hundreds of high-resolution plots.
Find other backends by entering %matplotlib -l
"""
if FilesOnly:
    mpl.use("cairo")

# Loop over edifiles:
n3d = 0
n2d = 0
n1d = 0
nel = 0
sit = 0

# fig  = plt.figure()
# fig.set_figwidth(16*cm)

dimlist = []
for filename in edi_files:
    sit = sit + 1
    print("reading data from: " + filename)
    name, ext = os.path.splitext(filename)
    file_i = filename

# Create MT object

    mt_obj = MT(file_i)
    lon = mt_obj.lon
    lat = mt_obj.lat
    elev = mt_obj.elev
    east = mt_obj.east
    north = mt_obj.north
    freq = mt_obj.Z.freq
# use the phase tensor to determine which frequencies are 1D/2D/3D
    dim = dimensionality(z_object=mt_obj.Z,
                         skew_threshold=5,
                         # threshold in skew angle (degrees) to determine if
                         # data are 3d
                         # threshold in phase ellipse eccentricity to determine
                         # if data are 2d (vs 1d)
                         eccentricity_threshold=0.1
                         )
    dims = np.vstack([freq,dim])
    print(np.shape(dims))
    np.savetxt(fname =name+"_dim.dat",fmt="%12.5f  %3i",X =dims.T)

    print("dimensionality:")
    nel_site = np.size(dim)
    n1d_site = sum(map(lambda x: x == 1, dim))
    n2d_site = sum(map(lambda x: x == 2, dim))
    n3d_site = sum(map(lambda x: x == 3, dim))

    _, sitn = os.path.split(name)
    dimlist.append([sitn, nel_site, n1d_site, n2d_site, n3d_site])

    nel = nel + np.size(dim)
    n1d = n1d + n1d_site
    n2d = n2d + n2d_site
    n3d = n3d + n3d_site

    strikeplot = PlotStrike(fn_list=[file_i],
                            plot_type=1,
                            plot_tipper='yr')
    #
    strikeplot.save_plot(name+"_Strikes.png",
                          file_format='.png',
                          fig_dpi=600)
    print("  number of undetermined elements = " +
          str(nel_site - n1d_site - n2d_site- n3d_site) + "\n")
    print("  number of 1-D elements = " + str(n1d_site) +
          "  (" + str(round(100 * n1d_site / nel)) + "%)")
    print("  number of 2-D elements = " + str(n2d_site) +
          "  (" + str(round(100 * n2d_site / nel)) + "%)")
    print("  number of 3-D elements = " + str(n3d_site) +
          "  (" + str(round(100 * n3d_site / nel)) + "%)")


strikeplot = PlotStrike(fn_list=edi_files,
                        plot_type=1,
                        plot_tipper='yr')
#
strikeplot.save_plot(edi_dir+"AllStrikes.png",
                      file_format='png',
                      fig_dpi=600)

print("\n\n\n")
print("number of sites = " + str(sit))
print("total number of elements = " + str(nel))
print("  number of undetermined elements = " +
      str(nel - n1d - n2d - n3d) + "\n")
print("  number of 1-D elements = " + str(n1d) +
      "  (" + str(round(100 * n1d / nel)) + "%)")
print("  number of 2-D elements = " + str(n2d) +
      "  (" + str(round(100 * n2d / nel)) + "%)")
print("  number of 3-D elements = " + str(n3d) +
      "  (" + str(round(100 * n3d / nel)) + "%)")
dimlist.append(["all_sites", nel,
                round(100*n1d/nel),
                round(100*n2d/nel),
                round(100*n3d/nel)])

with open(DimFile, "w") as f:
    sites = csv.writer(f, delimiter = " ")
    sites.writerow(["Sitename", "Ntot", "N1d%", "N2d%", "N3d%"])
    sites.writerow([ns, " ", " "])

    for item in dimlist:
        sites.writerow(item)

    print('Done')
